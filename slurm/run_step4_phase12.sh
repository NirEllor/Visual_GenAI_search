#!/bin/bash
# Step 4 Phase 1+2 — Evaluate Phase 1+2 teachers and students.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
  --mem=40G -c2 --time=1-00 --gres=gpu:1 \
  --mail-type=ALL --mail-user="$EMAIL" \
  --job-name=step4_p12_eval \
  --wrap "bash -c '$RUN python step4_evaluate.py --generate --exp-name ae_kl_lpips_only_v1_phase12; $RUN python step4_evaluate.py --decode --exp-name ae_kl_lpips_only_v1_phase12; $RUN python step4_evaluate.py --metrics --exp-name ae_kl_lpips_only_v1_phase12; $RUN python step4_evaluate.py --plot --exp-name ae_kl_lpips_only_v1_phase12'" \
  | awk '{print $NF}')

echo "  Submitted step4 Phase12 → Job $JOB"
echo "step4_phase12 job ID: $JOB"
