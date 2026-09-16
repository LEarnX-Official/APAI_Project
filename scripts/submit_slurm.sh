#!/usr/bin/env bash
#SBATCH --job-name=apai-maml
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# --- GPU selection -----------------------------------------------------------
# Adjust to match the cluster. Two common spellings, pick one:
#   #SBATCH --partition=gpu
#   #SBATCH --gres=gpu:p40:1
#   #SBATCH --nodelist=gpunode2
#SBATCH --gres=gpu:1
# -----------------------------------------------------------------------------
#
# Usage on the cluster:
#   sbatch scripts/submit_slurm.sh smoke      # quick environment check first
#   sbatch scripts/submit_slurm.sh all        # baselines + MAML + ablations
#   sbatch scripts/submit_slurm.sh maml
#
# Run the smoke stage before anything long: it exercises every code path in
# about a minute and fails fast on a bad environment, rather than burning
# allocation to discover a missing package at hour three.

set -euo pipefail

STAGE="${1:-smoke}"
cd "$(dirname "$0")/.."
mkdir -p logs results
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "=== job ${SLURM_JOB_ID:-local} stage=$STAGE ==="
echo "host: $(hostname)"
echo "date: $(date -Is)"

# Cluster software stacks vary; load modules if the system provides them.
if command -v module >/dev/null 2>&1; then
    module load cuda 2>/dev/null || true
    module load python 2>/dev/null || true
fi

echo "--- environment ---"
python3 -c "import torch; print('torch', torch.__version__);
print('cuda available:', torch.cuda.is_available());
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# The P40 is Pascal: no Tensor Cores, weak fp16. Keep everything in fp32 and do
# not enable AMP here, or the job runs slower than it needs to.

case "$STAGE" in
  smoke)
    bash scripts/smoke_test.sh
    ;;
  baseline)
    python3 src/train_baseline.py --model simple_cnn
    python3 src/train_baseline.py --model convnet4
    ;;
  maml)
    python3 src/train_maml.py
    ;;
  ablations)
    python3 src/train_maml.py --k-shot 1 --tag maml_5way_1shot
    python3 src/train_maml.py --inner-steps 1 --tag maml_inner1
    python3 src/train_maml.py --second-order --tag maml_second_order
    ;;
  all)
    python3 src/train_baseline.py --model simple_cnn
    python3 src/train_baseline.py --model convnet4
    python3 src/train_maml.py
    python3 src/train_maml.py --k-shot 1 --tag maml_5way_1shot
    python3 src/train_maml.py --inner-steps 1 --tag maml_inner1
    python3 src/train_maml.py --second-order --tag maml_second_order
    ;;
  *)
    echo "unknown stage '$STAGE' (smoke|baseline|maml|ablations|all)" >&2
    exit 2
    ;;
esac

python3 scripts/make_report.py
echo "=== done $(date -Is) ==="
