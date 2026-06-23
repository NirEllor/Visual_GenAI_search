#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

METHOD="${1:-pca}"
N_SAMPLES="${2:-3000}"
shift 2

DEP_FLAG=""
DIMS=()
SIZES=()

# Usage examples:
#   bash slurm/run_analyze_student_latents.sh pca 3000
#   bash slurm/run_analyze_student_latents.sh pca 3000 512 1024
#   bash slurm/run_analyze_student_latents.sh all 3000 512 1024 afterok:123
#
# By default:
#   dims  = 64 128 256 384 512 1024
#   sizes = 50000 100000 150000 200000

for arg in "$@"; do
  if [[ "$arg" == afterok:* ]]; then
    DEP_FLAG="--dependency=$arg"
  else
    DIMS+=("$arg")
  fi
done

if [ ${#DIMS[@]} -eq 0 ]; then
  DIMS=(64 128 256 384 512 1024)
fi

SIZES=(50000 100000 150000 200000)

JOB=$(sbatch $DEP_FLAG $NODE_ARGS \
  --mem=100G \
  -c4 \
  --time=03:00:00 \
  --gres=gpu:0 \
  --mail-type=ALL \
  --mail-user="$EMAIL" \
  --job-name=student_latent_${METHOD} \
  --wrap "bash -c '$RUN python3 analyze_student_latents.py --method $METHOD --n-samples $N_SAMPLES --dims ${DIMS[*]} --sizes ${SIZES[*]} --include-teacher'" \
  | awk '{print $NF}')

echo "analyze_student_latents job IDs: $JOB"