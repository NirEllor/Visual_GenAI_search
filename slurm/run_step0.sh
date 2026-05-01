#!/bin/bash
# Step 0 — Train 6 ConvAutoencoders from scratch (one per latent dim).
# Usage:
#   bash slurm/run_step0.sh                        # no dependency
#   bash slurm/run_step0.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"   # project root

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG \
    --mem=30G -c2 --time=0-12 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=step0_ae_d${DIM} \
    --wrap "bash -c '$RUN python step0_train_autoencoder.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "  Submitted step0 dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "step0 job IDs: ${IDS[*]}"
