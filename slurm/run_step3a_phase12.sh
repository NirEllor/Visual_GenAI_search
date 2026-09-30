#!/bin/bash
# Step 3a Phase 1+2 — Generate synthetic datasets from Phase 1+2 teachers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=40G -c2 --time=1-00 --gres=gpu:1 \
    --mail-type=ALL --mail-user="$EMAIL" \
    --job-name=step3a_p12_d${DIM} \
    --wrap "bash -c '$RUN python step3a_generate.py --dim $DIM --exp-name ae_kl_lpips_only_v1_phase12'" \
    | awk '{print $NF}')
  echo "  Submitted step3a Phase12 dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "step3a_phase12 job IDs: ${IDS[*]}"
