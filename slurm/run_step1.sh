#!/bin/bash
# Step 1 — Encode 50k CIFAR-10 images through each AE and save latents.
# Usage:
#   bash slurm/run_step1.sh                        # no dependency
#   bash slurm/run_step1.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG \
    --mem=30G -c2 --time=0-04 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=step1_lat_d${DIM} \
    --wrap "bash -c '$RUN python step1_extract_latents.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "  Submitted step1 dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "step1 job IDs: ${IDS[*]}"
