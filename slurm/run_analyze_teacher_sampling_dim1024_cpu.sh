#!/bin/bash
# Analyze teacher sampling distribution for dim=1024 (CPU version).
# Uses CPU instead of GPU due to CUDA compatibility issues.
#
# Usage:
#   bash slurm/run_analyze_teacher_sampling_dim1024_cpu.sh
#   bash slurm/run_analyze_teacher_sampling_dim1024_cpu.sh afterok:JID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

DIM=1024

JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
  --mem=128G \
  -c16 \
  --time=12:00:00 \
  --mail-type=ALL \
  --mail-user="$EMAIL" \
  --job-name=teacher_sample_d${DIM}_cpu \
  --wrap "bash -c '$RUN python analyze_teacher_sampling.py --dim $DIM --steps 50 100 200 --n-samples 5000 --batch-size 256'" \
  | awk '{print $NF}')

echo "Submitted teacher sampling analysis dim=$DIM (CPU) → Job $JOB"
echo "Monitor with: squeue -u ellorw.nir && tail -f slurm-${JOB}.out"
