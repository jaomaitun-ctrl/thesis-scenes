#!/usr/bin/env bash
# Run a Python script with SOFA's interpreter and the ViscoOgden plugin on the plugin path.
#   plugins/ViscoOgden/run.sh tests/homogeneous_test.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFA_SRC="${SOFA_SRC:-$HOME/Developer/sofa}"
ENV="$SOFA_SRC/.pixi/envs/supported-plugins"
export SOFA_ROOT="$ENV"
export SOFA_PLUGIN_PATH="$HERE/build${SOFA_PLUGIN_PATH:+:$SOFA_PLUGIN_PATH}"
export PYTHONPATH="$ENV/plugins/SofaPython3/lib/python3/site-packages:${PYTHONPATH:-}"
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SOFA_SRC"
exec pixi run -e supported-plugins "$ENV/bin/python-sofa" "$HERE/$1" "${@:2}"
