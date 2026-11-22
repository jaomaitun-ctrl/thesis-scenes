#!/usr/bin/env bash
# Linux server variant of run.sh: plain env python can import Sofa here (python-sofa is macOS-only).
set -euo pipefail
E=/srv/train/sofa/.pixi/envs/supported-plugins
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SOFA_ROOT="$E"
export KMP_DUPLICATE_LIB_OK=TRUE
export LD_LIBRARY_PATH="$E/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$E/plugins/SofaPython3/lib/python3/site-packages:$HERE/.pydeps:$HERE:${PYTHONPATH:-}"
cd "$HERE"
exec "$E/bin/python" "$HERE/$1" "${@:2}"
