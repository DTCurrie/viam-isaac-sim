#!/bin/bash
# Thin wrapper: creates a sim machine for a real machine's config, end to end.
# See `tools/create_sim_machine.py --help` for the full contract.
#
# `--open-livestream` idempotently opens the livestream's ports (TCP 49100 for
# WebRTC signaling, UDP 47998 for media) before handing off to
# create_sim_machine.py: a firewall rule ingress to the `viam-isaac-sim`
# network tag, then (once the instance exists) that tag applied to it with
# `gcloud compute instances add-tags`. The tag is applied after creation,
# not passed to `gcloud compute instances create` itself, because that
# command is built entirely inside create_sim_machine.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python"
CREATE_SIM_MACHINE="$SCRIPT_DIR/../tools/create_sim_machine.py"

LIVESTREAM_TAG="viam-isaac-sim"
LIVESTREAM_FIREWALL_RULE="viam-isaac-sim-livestream"

open_livestream=0
name=""
zone="us-central1-a"
project=""
dry_run=0
forward_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --open-livestream)
            open_livestream=1
            shift
            ;;
        --name)
            name="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --zone)
            zone="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --project)
            project="$2"
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --dry-run)
            dry_run=1
            forward_args+=("$1")
            shift
            ;;
        *)
            forward_args+=("$1")
            shift
            ;;
    esac
done

project_args=()
[ -n "$project" ] && project_args=(--project "$project")

open_livestream_firewall() {
    if gcloud compute firewall-rules describe "$LIVESTREAM_FIREWALL_RULE" "${project_args[@]}" >/dev/null 2>&1; then
        return
    fi
    gcloud compute firewall-rules create "$LIVESTREAM_FIREWALL_RULE" \
        "${project_args[@]}" \
        --direction INGRESS \
        --action ALLOW \
        --rules tcp:49100,udp:47998 \
        --target-tags "$LIVESTREAM_TAG"
}

if [ "$open_livestream" -eq 1 ] && [ "$dry_run" -eq 0 ]; then
    open_livestream_firewall
fi

set +e
"$PYTHON" "$CREATE_SIM_MACHINE" "${forward_args[@]}"
status=$?
set -e

if [ "$open_livestream" -eq 1 ] && [ "$dry_run" -eq 0 ] && [ "$status" -eq 0 ]; then
    if [ -z "$name" ]; then
        echo "create-sim-machine: --open-livestream needs --name to tag the instance" >&2
    else
        gcloud compute instances add-tags "$name" --zone "$zone" "${project_args[@]}" --tags "$LIVESTREAM_TAG"
    fi
fi

exit "$status"
