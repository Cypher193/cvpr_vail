#!/usr/bin/env bash
# Day 1: one clip per generator, then one tracked clip. Check: runs complete, wall time per clip (G7 early look),
# activation shapes, and that Wan-VACE reproduces its input frames (G9 early look: open runs/b1/vace13/*/video.mp4).
set -uo pipefail          # keep going: one failing model must not hide the others
export HF_HOME=${HF_HOME:-/workspace/hf}
for m in vace13 ltx2b cosmos2b; do
  python -m pilot.generate --model "$m" --exp b1 --limit 1 --seeds 0
done
python -m pilot.generate --model cosmos2b --exp b1 --limit 1 --seeds 0 --res 480x832   # compare with 720p
python - <<'EOF'
import json, glob, numpy as np
for d in sorted(glob.glob("runs/b1/*/*_s0")):
    meta = json.load(open(f"{d}/meta.json"))
    acts = np.load(f"{d}/acts.npz")
    print(d.split("/")[-2], f"{meta['wall_s']} s/clip", f"peak {meta.get('peak_gb')} GB", meta["cond_seconds"], "s input",
          {k: acts[k].shape for k in acts.files})
EOF
python scripts/sync.py push
