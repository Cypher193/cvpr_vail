#!/usr/bin/env bash
# P11 timing for the two models used only for budgeting: Cosmos-Predict2.5-14B (diffusers branch, same code as 2B)
# and Cosmos 3 Nano (needs diffusers main: own venv). Frees the Wan caches first (their runs are done by now).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/pod_env.sh
rm -rf "$HF_HOME"/hub/models--Wan-AI--* && echo "freed Wan caches"; df -h / | tail -1
timeout 2400 python -m pilot.generate --model cosmos14b --exp b1 --limit 1 --seeds 0 --no-capture; echo "cosmos14b exit $?"
rm -rf "$HF_HOME"/hub/models--nvidia--Cosmos-Predict2.5-14B
if [ ! -x /workspace/venv-nano/bin/python ]; then
  uv venv --python 3.13 --seed --managed-python /workspace/venv-nano
  uv pip install --python /workspace/venv-nano "diffusers @ git+https://github.com/huggingface/diffusers.git" \
    accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg torch torchvision transformers
fi
timeout 2400 /workspace/venv-nano/bin/python scripts/time_nano.py; echo "cosmos3_nano exit $?"
