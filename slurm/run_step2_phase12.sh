#!/bin/bash
# Step 2 Phase 1+2 — Train 6 teacher flow matching models with improved hyperparameters.
# Phase 1 fixes: lower weight_decay (1e-5), higher grad_clip (5.0), lower EMA decay (0.99)
# Phase 2 fixes: per-layer grad clipping, scaled LR per dim (3.0e-4 → 3.2e-4 for dim=1024)
#
# Usage:
#   bash slurm/run_step2_phase12.sh                        # no dependency
#   bash slurm/run_step2_phase12.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=30G -c2 --time=2-00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=step2_p12_d${DIM} \
    --wrap "bash -c '$RUN python step2_train_teachers.py --dim $DIM --exp-name ae_kl_lpips_only_v1_phase12'" \
    | awk '{print $NF}')
  echo "  Submitted step2 Phase1+2 dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "step2_phase12 job IDs: ${IDS[*]}"
echo "Monitor: squeue -u ellorw.nir -o \"%.10i %.20j %.8T %.10M\""
