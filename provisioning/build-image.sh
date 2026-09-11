#!/usr/bin/env bash
# Builds the `viam-isaac-sim` GCP image: a GPU instance with the NVIDIA driver,
# Python 3.11, and Isaac Sim baked into the venv `first_run.sh` looks for, a warm
# shader cache, and viam-agent installed and ready for a machine's credentials.
# `create-sim-machine.sh` launches instances from the resulting image.
#
# Usage: build-image.sh --project PROJECT [--zone ZONE] [--machine-type TYPE]
#   [--image-family FAMILY] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT=""
ZONE="us-central1-a"
MACHINE_TYPE="g2-standard-8"
IMAGE_FAMILY="viam-isaac-sim"
DRY_RUN=0

# Ubuntu 24.04's project and family name. `gcloud` is logged out in this
# environment so this could not be confirmed against the image list; corrected
# by the reference VM inventory (see provisioning/README.md "Pins") if wrong.
SOURCE_IMAGE_PROJECT="ubuntu-os-cloud"
SOURCE_IMAGE_FAMILY="ubuntu-2404-lts-amd64"

# Isaac Sim's pip package plus its cached extensions and the CUDA/driver stack
# run well past the 20GB Ubuntu default; 100GB leaves headroom for the venv,
# the extension cache, and viam-agent's own storage.
BOOT_DISK_SIZE_GB=100

DATA_DIR="/opt/viam-isaac-sim"
BUILDER_NAME=""
IMAGE_DATE=""

log() { echo "build-image: $*" >&2; }

usage() {
    cat <<'EOF'
Usage: build-image.sh --project PROJECT [--zone ZONE] [--machine-type TYPE]
  [--image-family FAMILY] [--dry-run]
EOF
}

# Prints every gcloud/SSH command it runs, so a dry run and a real run share
# exactly one source of truth for what happened.
run() {
    echo "+ $*"
    if [ "$DRY_RUN" -eq 0 ]; then
        "$@"
    fi
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --project)
                PROJECT="$2"
                shift 2
                ;;
            --zone)
                ZONE="$2"
                shift 2
                ;;
            --machine-type)
                MACHINE_TYPE="$2"
                shift 2
                ;;
            --image-family)
                IMAGE_FAMILY="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            *)
                log "unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
    if [ -z "$PROJECT" ] && [ "$DRY_RUN" -eq 0 ]; then
        log "--project is required"
        usage
        exit 1
    fi
    PROJECT="${PROJECT:-<project>}"
    BUILDER_NAME="viam-isaac-sim-builder-$(date +%Y%m%d%H%M%S)"
    IMAGE_DATE="$(date +%Y%m%d)"
}

create_builder() {
    run gcloud compute instances create "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --machine-type "$MACHINE_TYPE" \
        --image-family "$SOURCE_IMAGE_FAMILY" \
        --image-project "$SOURCE_IMAGE_PROJECT" \
        --accelerator "type=nvidia-l4,count=1" \
        --maintenance-policy TERMINATE \
        --boot-disk-size "${BOOT_DISK_SIZE_GB}GB"
}

wait_for_ssh() {
    run gcloud compute ssh "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --command "true" \
        --tries 30
}

copy_install_files() {
    run gcloud compute ssh "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --command "mkdir -p tools fragments"
    run gcloud compute scp "$SCRIPT_DIR/../first_run.sh" "$SCRIPT_DIR/../requirements.txt" \
        "$BUILDER_NAME:~/" \
        --project "$PROJECT" \
        --zone "$ZONE"
    run gcloud compute scp --recurse "$SCRIPT_DIR/../src" \
        "$BUILDER_NAME:~/" \
        --project "$PROJECT" \
        --zone "$ZONE"
    run gcloud compute scp "$SCRIPT_DIR/../tools/warm_shader_cache.py" \
        "$BUILDER_NAME:~/tools/" \
        --project "$PROJECT" \
        --zone "$ZONE"
    run gcloud compute scp "$SCRIPT_DIR/../fragments/isaac-sim-block-sorting.json" \
        "$BUILDER_NAME:~/fragments/" \
        --project "$PROJECT" \
        --zone "$ZONE"
}

run_first_run() {
    run gcloud compute ssh "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --command "sudo VIAM_MODULE_DATA=$DATA_DIR bash first_run.sh"
}

reboot_and_confirm() {
    run gcloud compute instances reset "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE"
    wait_for_ssh
    # Re-runs the installer so its fast path (a venv where `import isaacsim`
    # already succeeds) confirms the install landed instead of re-downloading it.
    run_first_run
}

# Runs as root because viam-agent runs the module as root, and Kit's per-user
# caches (`~/.cache/ov`, `~/.nv/ComputeCache`, `~/.cache/nvidia/GLCache`) must
# land in root's home to be found there later. Runs after the reboot so the
# driver is active. Uses our own fragment rather than NVIDIA's generic
# `warmup.sh` so the compiled pipelines include our assets, props and camera
# render products, not just the base pipelines. Kit's portable cache lands
# under the venv at `isaacsim/kit/cache`. A driver or Isaac Sim bump
# invalidates the cache, so the image is rebuilt after either.
warm_shader_cache() {
    run gcloud compute ssh "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --command "sudo VIAM_MODULE_DATA=$DATA_DIR OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src $DATA_DIR/isaac-venv/bin/python tools/warm_shader_cache.py --fragment fragments/isaac-sim-block-sorting.json"
}

install_viam_agent() {
    # https://docs.viam.com/fleet/provision-devices/ ("Provision a device with
    # a preinstall script"): the preinstall script installs viam-agent without
    # requiring a viam-defaults.json in the working directory.
    run gcloud compute ssh "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --command "curl -O https://storage.googleapis.com/packages.viam.com/apps/viam-agent/preinstall.sh && chmod 755 preinstall.sh && sudo ./preinstall.sh"
}

stop_builder() {
    run gcloud compute instances stop "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE"
}

create_image() {
    run gcloud compute images create "$IMAGE_FAMILY-$IMAGE_DATE" \
        --project "$PROJECT" \
        --source-disk "$BUILDER_NAME" \
        --source-disk-zone "$ZONE" \
        --family "$IMAGE_FAMILY"
}

delete_builder() {
    run gcloud compute instances delete "$BUILDER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --quiet
}

main() {
    parse_args "$@"
    create_builder
    wait_for_ssh
    copy_install_files
    run_first_run
    reboot_and_confirm
    warm_shader_cache
    install_viam_agent
    stop_builder
    create_image
    delete_builder
    if [ "$DRY_RUN" -eq 1 ]; then
        log "dry run only; no gcloud commands were executed"
    fi
}

main "$@"
