#!/bin/bash
EMAIL="ellorwaizner.nir@mail.huji.ac.il"
VENV_JAX="source /cs/labs/raananf/ellorw.nir/venv_jax/bin/activate"
DIMS=(64 128 256 384)

# Step 4b — decode + FID (JAX venv, no GPU needed)
STEP4B_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch \
    --mem=30G -c1 --time=13-23 --gres=gpu:0 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step4b_dim${DIM} \
    --wrap "bash -c '$VENV_JAX; python3 step4_evaluate.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step4b dim=$DIM → Job ID: $JOB"
  STEP4B_IDS+=($JOB)
done

STEP4B_DEP="afterok:$(IFS=:; echo "${STEP4B_IDS[*]}")"

# Step 4c — plot
JOB_PLOT=$(sbatch --dependency=$STEP4B_DEP \
  --mem=8G -c1 --time=0-01 --gres=gpu:0 \
  --mail-type=ALL --mail-user=$EMAIL --job-name=step4_plot \
  --wrap "bash -c '$VENV_JAX; python3 step4_evaluate.py --plot-only'" \
  | awk '{print $NF}')
echo "Submitted step4_plot → Job ID: $JOB_PLOT"