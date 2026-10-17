#!/usr/bin/env bash
# Whole pilot on one fresh A100 pod: setup (outside the budget), then the budgeted run.
# Start detached:  setsid nohup bash scripts/pilot_pod.sh > /workspace/pilot.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/pod_env.sh
bash scripts/setup_runpod.sh > data/logs/setup.log 2>&1 || { echo "setup failed"; exit 1; }
python -c "from cosmos_guardrail import CosmosSafetyChecker; CosmosSafetyChecker()" > data/logs/guardrail.log 2>&1 \
  || { echo "guardrail failed (see data/logs/guardrail.log)"; exit 1; }
python scripts/run_pilot.py --budget-h 11.5
echo "PILOT POD DONE"
