#!/bin/bash
# Step 0b — Evaluate trained AEs: reconstructions, FID, and summary plot.
# Submits 6 per-dim eval jobs, then 1 CPU plot job that waits for all of them.
# Usage:
#   bash slurm/run_step0b.sh                        # no dependency
#   bash slurm/run_step0b.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

EVAL_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG \
    --mem=30G -c2 --time=0-04 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=step0b_eval_d${DIM} \
    --wrap "bash -c '$VENV && export TORCH_HOME=$TORCH_HOME && python3 step0b_eval_ae.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "  Submitted step0b eval dim=$DIM → Job $JOB"
  EVAL_IDS+=($JOB)
done

EVAL_DEP="afterok:$(IFS=:; echo "${EVAL_IDS[*]}")"

JOB_PLOT=$(sbatch --dependency="$EVAL_DEP" \
  --mem=8G -c1 --time=0-01 --gres=gpu:0 \
  --mail-type=ALL --mail-user="$EMAIL" \
  --job-name=step0b_plot \
  --wrap "bash -c '$VENV && export TORCH_HOME=$TORCH_HOME && python3 step0b_eval_ae.py --plot-only'" \
  | awk '{print $NF}')
echo "  Submitted step0b plot → Job $JOB_PLOT"

ALL_IDS=("${EVAL_IDS[@]}" "$JOB_PLOT")
echo "step0b job IDs: ${ALL_IDS[*]}"
