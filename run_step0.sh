#!/bin/bash
EMAIL="ellorwaizner.nir@mail.huji.ac.il"
VENV_JAX="source /cs/labs/raananf/ellorw.nir/venv_jax/bin/activate"
DIMS=(8 16 32)

STEP0_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step0_dim${DIM} \
    --wrap "bash -c '$VENV_JAX; python3 step0_train_autoencoder.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step0 dim=$DIM → Job ID: $JOB"
  STEP0_IDS+=($JOB)
done

echo "Pipeline submitted: ${STEP0_IDS[*]}"