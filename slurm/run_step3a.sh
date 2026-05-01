#!/bin/bash
# Step 3a — Generate synthetic x_0 datasets + trajectory files from each teacher.
# Produces 4 dataset sizes (250k/500k/1M/2M) and 50k trajectories per dim.
# Usage:
#   bash slurm/run_step3a.sh                        # no dependency
#   bash slurm/run_step3a.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG \
    --mem=40G -c2 --time=0-12 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=step3a_gen_d${DIM} \
    --wrap "bash -c '$VENV && python3 step3a_generate.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "  Submitted step3a dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "step3a job IDs: ${IDS[*]}"
