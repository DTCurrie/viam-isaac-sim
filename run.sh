#!/usr/bin/env bash
# Entrypoint for the viam-isaac-sim module.
#
# Isaac Sim's python API only works inside Isaac Sim's own python
# environment, so we prefer that interpreter when we can find it:
#   1. $ISAAC_PYTHON             - explicit path to a python executable
#   2. $ISAAC_SIM_PATH/python.sh - standard Isaac Sim install layout
#   3. the venv first_run.sh created (marker in $VIAM_MODULE_DATA)
#   4. python3                   - fallback (mock mode only)
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${VIAM_MODULE_DATA:-/opt/viam-isaac-sim}"
MARKER="$DATA_DIR/isaac_python"

if [ -n "${ISAAC_PYTHON:-}" ]; then
    PY="$ISAAC_PYTHON"
elif [ -n "${ISAAC_SIM_PATH:-}" ] && [ -x "$ISAAC_SIM_PATH/python.sh" ]; then
    PY="$ISAAC_SIM_PATH/python.sh"
elif [ -f "$MARKER" ] && [ -x "$(cat "$MARKER")" ]; then
    PY="$(cat "$MARKER")"
elif [ -x "$DATA_DIR/isaac-venv/bin/python" ]; then
    PY="$DATA_DIR/isaac-venv/bin/python"
elif [ -x "$HOME/isaacsim/python.sh" ]; then
    PY="$HOME/isaacsim/python.sh"
else
    PY="python3"
    echo "viam-isaac-sim: Isaac Sim python not found (set ISAAC_SIM_PATH); only mock mode will work" >&2
fi

# pip-installed isaac sim prompts for the EULA on first boot and refuses to
# run as root without these
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-yes}
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export OMNI_KIT_ALLOW_ROOT=${OMNI_KIT_ALLOW_ROOT:-1}

# One-time dependency install into whichever interpreter we're using.
if ! "$PY" -c "import viam" >/dev/null 2>&1; then
    echo "viam-isaac-sim: installing python dependencies..." >&2
    "$PY" -m pip install -r requirements.txt >&2
fi

exec "$PY" src/main.py "$@"
