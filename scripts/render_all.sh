#!/usr/bin/env bash
# Render every pilot scene with Blender (bpy 5.2 needs Python 3.13, so it gets its own venv via uv).
# About 12,500 frames. Re-running skips clips that already exist. Works on RunPod (Linux) or locally.
set -euo pipefail
# The Docker image ships Blender in /opt/bpy (BPY_PYTHON). Without the image, build a local venv.
PY=${BPY_PYTHON:-}
if [ -z "$PY" ]; then
  if [ ! -d .venv-bpy ]; then
    if command -v apt-get >/dev/null; then       # libraries the bpy wheel needs on a headless Linux pod
      apt-get update -qq && apt-get install -y -qq libxrender1 libxi6 libxkbcommon0 libxxf86vm1 libxfixes3         libsm6 libice6 libgl1 libegl1 >/dev/null
    fi
    command -v uv >/dev/null || pip install -q uv
    uv venv --python 3.13 .venv-bpy
    uv pip install --python .venv-bpy bpy==5.2.2 numpy pyyaml
  fi
  PY=.venv-bpy/bin/python
  [ -x "$PY" ] || PY=.venv-bpy/Scripts/python
fi
python scripts/sync.py pull || true          # resume: finished clips from an earlier pod are skipped
python -m pilot.scenes --out data/scenes
mkdir -p data/logs
python scripts/sync.py push --every 600 > data/logs/sync_render.log 2>&1 &   # push finished clips every 10 min
SYNC_PID=$!
# Four independent jobs in parallel (one process does not fill the GPU); logs in data/logs/.
#   full 24 fps clips: tracker validation (P0) and rendered-vs-real comparison
#   48 fps input windows: exact input frames at 16 fps (every 3rd) and 24 fps (every 2nd)
#   C2-g continuations at Cosmos's native 16 fps (93 frames)
pids=()
$PY -m pilot.render data/scenes/kmain --out data/renders/kmain --gpu > data/logs/render_kmain.log 2>&1 & pids+=($!)
( for s in kmain kgrav kratio; do
    $PY -m pilot.render data/scenes/$s --out data/renders/$s --gpu --fps 48 --seconds 0.5 --suffix _in48
  done ) > data/logs/render_inputs.log 2>&1 & pids+=($!)
# kcont split in two halves so the longest job does not set the wall time alone
mkdir -p data/scenes/kcont_a data/scenes/kcont_b
for f in data/scenes/kcont/kcont_0[0-4]_*.json; do cp "$f" data/scenes/kcont_a/; done
for f in data/scenes/kcont/kcont_0[5-9]_*.json; do cp "$f" data/scenes/kcont_b/; done
for h in a b; do
  $PY -m pilot.render data/scenes/kcont_$h --out data/renders/kcont --gpu --fps 16 --seconds 5.8125     > data/logs/render_kcont_$h.log 2>&1 & pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
rm -rf data/scenes/kcont_a data/scenes/kcont_b
echo "clips: $(ls data/renders/*/*.mp4 | wc -l) (expected 30 full + 70 inputs + 70 kcont = 170)"
kill $SYNC_PID 2>/dev/null || true
python scripts/sync.py push
exit $fail
