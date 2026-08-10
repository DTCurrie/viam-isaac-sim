#!/usr/bin/env bash
# One-time machine setup, run automatically by viam-server when the module is
# first installed (meta.json "first_run").
#
# Turns a standard Ubuntu 22.04/24.04 x86_64 machine into one that can run
# Isaac Sim:
#   - system libraries kit needs (vulkan, GL)
#   - NVIDIA driver if none is present (a reboot may be needed after)
#   - the python version Isaac Sim requires (deadsnakes PPA on 24.04)
#   - Isaac Sim itself, pip-installed into a venv under the module data dir
#   - this module's python deps into the same venv
#
# run.sh finds the result via the marker file written at the end, so no
# ISAAC_SIM_PATH / ISAAC_PYTHON configuration is needed.
#
# The isaacsim download is large (10GB+); if it exceeds viam-server's default
# first_run timeout, set "first_run_timeout": "2h0m0s" on the module config.
set -uo pipefail

log() { echo "viam-isaac-sim first_run: $*"; }

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    log "not linux/x86_64 - nothing to install (mock mode still works)"
    exit 0
fi

if [ -n "${ISAAC_PYTHON:-}" ] || [ -n "${ISAAC_SIM_PATH:-}" ]; then
    log "ISAAC_PYTHON/ISAAC_SIM_PATH already configured - skipping install"
    exit 0
fi

DATA_DIR="${VIAM_MODULE_DATA:-/opt/viam-isaac-sim}"
VENV="$DATA_DIR/isaac-venv"
MARKER="$DATA_DIR/isaac_python"
MODULE_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DATA_DIR"

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import isaacsim" >/dev/null 2>&1; then
    log "isaac sim already installed at $VENV"
    echo "$VENV/bin/python" > "$MARKER"
    exit 0
fi

SUDO=""
if [ "$(id -u)" != "0" ]; then
    if sudo -n true 2>/dev/null; then
        SUDO="sudo -n"
    else
        log "WARNING: not root and no passwordless sudo; skipping apt steps"
    fi
fi

# ---------------------------------------------------------------------------
# pick isaac sim version by ubuntu release (it dictates the python version)
# ---------------------------------------------------------------------------
. /etc/os-release 2>/dev/null || true
UBUNTU="${VERSION_ID:-unknown}"
NEED_DEADSNAKES=0
case "$UBUNTU" in
    22.04)
        PYBIN=python3.10
        ISAAC_VERSION=4.5.0
        ;;
    24.04)
        PYBIN=python3.11
        ISAAC_VERSION=5.0.0
        NEED_DEADSNAKES=1
        ;;
    *)
        log "WARNING: untested distro ($PRETTY_NAME); trying python3.11 + isaac 5.0.0"
        PYBIN=python3.11
        ISAAC_VERSION=5.0.0
        NEED_DEADSNAKES=1
        ;;
esac
log "ubuntu $UBUNTU -> isaac sim $ISAAC_VERSION on $PYBIN"

# ---------------------------------------------------------------------------
# apt: system libs, python, gpu driver
# ---------------------------------------------------------------------------
if [ "$(id -u)" = "0" ] || [ -n "$SUDO" ]; then
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update -qq || log "WARNING: apt-get update failed; continuing"
    $SUDO apt-get install -y -qq \
        software-properties-common curl ca-certificates \
        libvulkan1 vulkan-tools libglu1-mesa libegl1 libgomp1 libxt6 libxrandr2 \
        || log "WARNING: some system libraries failed to install"

    if [ "$NEED_DEADSNAKES" = "1" ] && ! command -v "$PYBIN" >/dev/null 2>&1; then
        log "adding deadsnakes PPA for $PYBIN"
        $SUDO add-apt-repository -y ppa:deadsnakes/ppa || log "WARNING: deadsnakes PPA failed"
        $SUDO apt-get update -qq || true
    fi
    $SUDO apt-get install -y -qq "$PYBIN" "$PYBIN-venv" "$PYBIN-dev" \
        || log "WARNING: installing $PYBIN failed"

    # Isaac Sim only supports validated driver branches; the R590/595 branch
    # is known to crash the RTX renderer (isaac-sim/IsaacSim#537, #643).
    # 580.x is the validated branch for Isaac 5.0 on Linux.
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "no NVIDIA driver found - installing the validated 580 branch"
        $SUDO apt-get install -y -qq nvidia-driver-580 \
            || $SUDO apt-get install -y -qq nvidia-driver-580-open \
            || log "WARNING: driver install failed; install the 580-branch NVIDIA driver manually"
        log "NOTE: a REBOOT is likely required before the GPU is usable"
    else
        DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        log "NVIDIA driver present: ${DRIVER_VER:-unknown}"
        DRIVER_MAJOR="${DRIVER_VER%%.*}"
        case "$DRIVER_MAJOR" in
            ''|*[!0-9]*) ;;
            *)
                if [ "$DRIVER_MAJOR" -ge 590 ]; then
                    log "WARNING: driver $DRIVER_VER is NOT validated for isaac sim and is"
                    log "WARNING: known to crash the RTX renderer (librtx.scenedb crash) and"
                    log "WARNING: break CUDA init (cuDeviceGetUuid). Downgrade to the 580 branch:"
                    log "WARNING:   sudo apt-get install -y nvidia-driver-580 && sudo reboot"
                fi
                ;;
        esac
    fi
fi

if ! command -v "$PYBIN" >/dev/null 2>&1; then
    log "ERROR: $PYBIN is not available; cannot install isaac sim"
    log "install it manually, or set ISAAC_SIM_PATH to an existing isaac sim install"
    exit 1
fi

# ---------------------------------------------------------------------------
# isaac sim via pip into a dedicated venv
# ---------------------------------------------------------------------------
log "creating venv at $VENV"
"$PYBIN" -m venv "$VENV" || { log "ERROR: venv creation failed"; exit 1; }
"$VENV/bin/pip" install --upgrade pip -q

log "installing isaacsim==$ISAAC_VERSION (this downloads 10GB+, be patient)"
if ! "$VENV/bin/pip" install \
        "isaacsim[all,extscache]==$ISAAC_VERSION" \
        --extra-index-url https://pypi.nvidia.com; then
    log "ERROR: isaac sim pip install failed"
    log "install isaac sim manually and set ISAAC_SIM_PATH, or use mock mode"
    exit 1
fi

log "installing module python dependencies"
"$VENV/bin/pip" install -q -r "$MODULE_DIR/requirements.txt" \
    || { log "ERROR: module dependency install failed"; exit 1; }

echo "$VENV/bin/python" > "$MARKER"
log "done - isaac python recorded at $MARKER"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "REMINDER: reboot to finish NVIDIA driver setup before starting the world"
fi
