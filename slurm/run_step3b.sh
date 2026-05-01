#!/bin/bash
# Step 3b — Distil 24 student models (6 dims × 4 dataset sizes).
# Each job trains one (dim, size) student independently; all use --load-to-ram.
# Max dataset RAM: dim=1024 n=2M → ~8.4 GB, well within 30G allocation.
# Usage:
#   bash slurm/run_step3b.sh                        # no dependency
#   bash slurm/run_step3b.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()
for DIM in "${DIMS[@]}"; do
  for SIZE in "${SIZES[@]}"; do
    JOB=$(sbatch $DEP_FLAG \
      --mem=30G -c2 --time=1-00 --gres=gpu:1 \
      --mail-type=ALL --mail-user="$EMAIL" \
      --job-name=step3b_stu_d${DIM}_n${SIZE} \
      --wrap "bash -c '$VENV && python3 step3b_distill.py --dim $DIM --size $SIZE --load-to-ram'" \
      | awk '{print $NF}')
    echo "  Submitted step3b dim=$DIM size=$SIZE → Job $JOB"
    IDS+=($JOB)
  done
done

echo "step3b job IDs: ${IDS[*]}"
