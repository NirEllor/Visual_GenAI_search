#!/bin/bash
# Toy Flow Matching sanity check — runs all latent dims sequentially.
#
# Usage:
#   bash slurm/run_toy_flow_sanity.sh
#   bash slurm/run_toy_flow_sanity.sh afterok:JID1:JID2:...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
  --mem=30G \
  -c2 \
  --time=08:00:00 \
  --gres=gpu:1 \
  --mail-type=ALL \
  --mail-user="$EMAIL" \
  --job-name=toy_fm_all_dims \
  --wrap "bash -c '$RUN python sanity_flow.py'" \
  | awk '{print $NF}')

echo "toy_flow_sanity job IDs: $JOB"