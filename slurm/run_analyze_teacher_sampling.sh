#!/bin/bash
# Analyze teacher sampling variance/collapse across Euler steps.
#
# Usage:
#   bash slurm/run_analyze_teacher_sampling.sh
#   bash slurm/run_analyze_teacher_sampling.sh afterok:JID1:JID2:...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

IDS=()

for DIM in 64 128 256 384 512 1024; do
  JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
    --mem=30G \
    -c2 \
    --time=04:00:00 \
    --gres=gpu:1 \
    --mail-type=ALL \
    --mail-user="$EMAIL" \
    --job-name=teacher_sample_d${DIM} \
    --wrap "bash -c '$RUN python analyze_teacher_sampling.py --dim $DIM --steps 50 100 200 400 --n-samples 10000 --batch-size 512'" \
    | awk '{print $NF}')

  echo "Submitted teacher sampling analysis dim=$DIM → Job $JOB"
  IDS+=($JOB)
done

echo "teacher_sampling_analysis job IDs: ${IDS[*]}"