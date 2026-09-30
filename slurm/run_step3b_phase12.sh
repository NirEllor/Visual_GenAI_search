#!/bin/bash
# Step 3b Phase 1+2 — Train students on Phase 1+2 synthetic data.

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
    --job-name=step3b_p12_d${DIM} \
    --wrap "bash -c '$RUN python step3b_distill.py --dim $DIM --exp-name ae_kl_lpips_only_v1_phase12'" \
    | awk '{print $NF}')
  echo "  Submitted step3b Phase12 dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "step3b_phase12 job IDs: ${IDS[*]}"
