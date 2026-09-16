#!/usr/bin/env bash
# Fast end-to-end check that every code path runs. Not a real experiment:
# the settings are far too small for meaningful numbers.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "=== data inspection ==="
python3 scripts/inspect_data.py

echo
echo "=== baseline (2 epochs, 10 batches) ==="
python3 src/train_baseline.py --model simple_cnn --epochs 2 \
    --limit-batches 10 --batch-size 64 --tag smoke_baseline

echo
echo "=== MAML (30 iterations) ==="
python3 src/train_maml.py --iterations 30 --eval-every 15 \
    --eval-episodes 20 --tag smoke_maml

echo
echo "=== report ==="
python3 scripts/make_report.py

echo
echo "smoke test passed"
