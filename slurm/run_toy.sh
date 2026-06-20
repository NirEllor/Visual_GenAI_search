#!/bin/bash
# Toy Flow Matching sanity check
#
# Usage:
#   bash slurm/run_toy_flow_sanity.sh
#   bash slurm/run_toy_flow_sanity.sh afterok:JID1:JID2:...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()

for DIM in 64 256 1024; do

  JOB=$(sbatch $DEP_FLAG \
    --mem=20G \
    -c2 \
    --time=04:00:00 \
    --gres=gpu:1 \
    --mail-type=ALL \
    --mail-user="$EMAIL" \
    --job-name=toy_fm_d${DIM} \
    --wrap "bash -c '$RUN python sanity_flow.py --dim $DIM'" \
    | awk '{print $NF}')

  echo "  Submitted toy flow sanity dim=$DIM → Job $JOB"
  IDS+=($JOB)

done

echo "toy_flow_sanity job IDs: ${IDS[*]}"