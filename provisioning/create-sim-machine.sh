#!/bin/bash
# Thin wrapper: creates a sim machine for a real machine's config, end to end.
# See `tools/create_sim_machine.py --help` for the full contract.
exec "$(dirname "$0")/../.venv/bin/python" "$(dirname "$0")/../tools/create_sim_machine.py" "$@"
