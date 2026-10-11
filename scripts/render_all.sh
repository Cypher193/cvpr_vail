#!/usr/bin/env bash
# Render every pilot scene with Blender (bpy 5.2, Python 3.13 venv). About 12,500 frames;
# ~2.2 s/frame on an RTX 4060 laptop (~7.5 h). Re-running skips clips that already exist.
#   python -m venv .venv-bpy && .venv-bpy/bin/pip install bpy==5.2.2 numpy pyyaml
set -euo pipefail
PY=${PY:-.venv-bpy/bin/python}
[ -x "$PY" ] || PY=.venv-bpy/Scripts/python
python -m pilot.scenes --out data/scenes
# full 24 fps clips: tracker validation (P0) and rendered-vs-real comparison
$PY -m pilot.render data/scenes/kmain --out data/renders/kmain --gpu
# 48 fps input windows: exact input frames at 16 fps (every 3rd) and 24 fps (every 2nd)
for s in kmain kgrav kratio; do
  $PY -m pilot.render data/scenes/$s --out data/renders/$s --gpu --fps 48 --seconds 0.5 --suffix _in48
done
# C2-g continuations at Cosmos's native 16 fps (93 frames)
$PY -m pilot.render data/scenes/kcont --out data/renders/kcont --gpu --fps 16 --seconds 5.8125
