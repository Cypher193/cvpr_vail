#!/usr/bin/env bash
# Render every pilot scene with Blender (bpy 5.2 needs Python 3.13, so it gets its own venv via uv).
# About 12,500 frames. Re-running skips clips that already exist. Works on RunPod (Linux) or locally.
set -euo pipefail
if [ ! -d .venv-bpy ]; then
  if command -v apt-get >/dev/null; then       # libraries the bpy wheel needs on a headless Linux pod
    apt-get update -qq && apt-get install -y -qq libxrender1 libxi6 libxkbcommon0 libxxf86vm1 libxfixes3 \
      libsm6 libice6 libgl1 libegl1 >/dev/null
  fi
  command -v uv >/dev/null || pip install -q uv
  uv venv --python 3.13 .venv-bpy
  uv pip install --python .venv-bpy bpy==5.2.2 numpy pyyaml
fi
PY=.venv-bpy/bin/python
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
