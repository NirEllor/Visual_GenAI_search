#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

METHOD="${1:-pca}"
N_SAMPLES="${2:-1000}"
shift 2

DEP_FLAG=""
DIMS=()

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

JOB=$(sbatch $DEP_FLAG \
  --mem=100G \
  -c4 \
  --time=03:00:00 \
  --gres=gpu:0 \
  --mail-type=FAIL,END \
  --mail-user="$EMAIL" \
  --job-name=viz_latents_${METHOD} \
  --wrap "bash -c '$RUN python3 visualize_latents.py --method $METHOD --n-samples $N_SAMPLES --dims ${DIMS[*]}'" \
  | awk '{print $NF}')

echo "visualize_latents job IDs: $JOB"