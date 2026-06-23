#!/bin/bash
# Gaussian latent baseline decode.
#
# Usage:
#   bash slurm/run_gaussian_baseline.sh
#   bash slurm/run_gaussian_baseline.sh afterok:JID1:JID2:...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()

for DIM in 128 256 512 1024; do
  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=30G \
    -c2 \
    --time=04:00:00 \
    --gres=gpu:1 \
    --mail-type=ALL \
    --mail-user="$EMAIL" \
    --job-name=gauss_base_d${DIM} \
    --wrap "bash -c '$RUN python gaussian_baseline_decode.py --dim $DIM'" \
        | awk '{print $NF}')

  echo "Submitted gaussian baseline dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "gaussian_baseline job IDs: ${IDS[*]}"