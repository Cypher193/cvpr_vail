#!/usr/bin/env bash
# Fresh RunPod pod (PyTorch image, A100 80 GB, network volume >= 150 GB mounted at /workspace).
# Set HF_TOKEN as a RunPod secret / env var. Never write the token into the repo.
set -euo pipefail
cd /workspace
[ -d cvpr_vail ] || git clone https://github.com/Cypher193/cvpr_vail.git
cd cvpr_vail
export HF_HOME=/workspace/hf
pip install -q -r requirements.txt
python - <<'EOF'
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
# ~100 GB in total; the text encoders are most of it
for repo, rev in [("nvidia/Cosmos-Predict2.5-2B", "diffusers/base/post-trained"),
                  ("Lightricks/LTX-Video-0.9.5", None),
                  ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", None),
                  ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", None),
                  ("facebook/sam2.1-hiera-large", None)]:
    print("downloading", repo, flush=True)
    snapshot_download(repo, revision=rev)
EOF
python -m pilot.metrics
python -m pilot.download
echo "Setup done. Copy data/scenes and data/renders here, then run scripts/smoke_test.sh"
