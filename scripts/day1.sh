#!/usr/bin/env bash
# Day 1 on a fresh pod: setup (weights) and rendering run in parallel, then smoke test and tracker check (G1a).
# Run detached: nohup bash scripts/day1.sh > /workspace/day1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/pod_env.sh
mkdir -p data/logs
bash scripts/setup_runpod.sh > data/logs/setup.log 2>&1 & SETUP=$!
bash scripts/render_all.sh > data/logs/render_all.log 2>&1 & RENDER=$!
wait $SETUP; echo "setup exit $?"; wait $RENDER; echo "render exit $?"
bash scripts/smoke_test.sh > data/logs/smoke.log 2>&1; echo "smoke exit $?"
python -m pilot.track validate data/renders/kmain --scenes data/scenes/kmain > data/logs/p0_renders.log 2>&1; echo "p0 renders exit $?"
python -m pilot.track qval > data/logs/p0_qval.log 2>&1; echo "p0 qval exit $?"
python scripts/sync.py push; echo "DAY1 DONE"
