#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

METHOD="${1:-pca}"
N_SAMPLES="${2:-1000}"
DIMS="${@:3}"

if [ -z "$DIMS" ]; then
  DIMS="128 256 384"
fi

JOB=$(sbatch \
  --mem=64G \
  -c4 \
  --time=03:00:00 \
  --gres=gpu:0 \
  --job-name=viz_latents_${METHOD} \
  --wrap "bash -c '$RUN python3 visualize_latents.py --method $METHOD --n-samples $N_SAMPLES --dims $DIMS'" \
  | awk '{print $NF}')

echo "visualize_latents job IDs: $JOB"