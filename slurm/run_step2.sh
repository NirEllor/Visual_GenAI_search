#!/bin/bash
# Step 2 — Train 6 teacher flow matching models (one per latent dim).
# Usage:
#   bash slurm/run_step2.sh                        # no dependency
#   bash slurm/run_step2.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG \
    --mem=30G -c2 --time=1-00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=step2_tea_d${DIM} \
    --wrap "bash -c '$VENV && python3 step2_train_teachers.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "  Submitted step2 dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "step2 job IDs: ${IDS[*]}"
