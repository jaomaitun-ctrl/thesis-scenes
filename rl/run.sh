#!/usr/bin/env bash
# Run a Python script with SOFA's interpreter (python-sofa) from the pixi env, so that
# `import Sofa` works in-process.  Usage:  rl/run.sh rl/check_env.py [args]
set -euo pipefail
SOFA_SRC="${SOFA_SRC:-$HOME/Developer/sofa}"
ENV="$SOFA_SRC/.pixi/envs/supported-plugins"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SOFA_ROOT="$ENV"
# torch (in .pydeps) ships its own libomp and the SOFA env has another; allow both
# (torch runs single-threaded in each env process, see train_ppo.py)
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="$ENV/plugins/SofaPython3/lib/python3/site-packages:$HERE/.pydeps:$HERE:${PYTHONPATH:-}"
export PATH="$HOME/.pixi/bin:$PATH"
cd "$SOFA_SRC"
exec pixi run -e supported-plugins "$ENV/bin/python-sofa" "$HERE/$1" "${@:2}"
