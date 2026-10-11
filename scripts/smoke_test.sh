#!/usr/bin/env bash
# Day 1: one clip per generator, then one tracked clip. Check: runs complete, wall time per clip (G7 early look),
# activation shapes, and that Wan-VACE reproduces its input frames (G9 early look: open runs/b1/vace13/*/video.mp4).
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
for m in vace13 ltx2b cosmos2b; do
  python -m pilot.generate --model "$m" --exp b1 --limit 1 --seeds 0
done
python - <<'EOF'
import json, glob, numpy as np
for d in sorted(glob.glob("runs/b1/*/*_s0")):
    meta = json.load(open(f"{d}/meta.json"))
    acts = np.load(f"{d}/acts.npz")
    print(meta["model"], f"{meta['wall_s']} s/clip", meta["cond_seconds"], "s input",
          {k: acts[k].shape for k in acts.files})
EOF
