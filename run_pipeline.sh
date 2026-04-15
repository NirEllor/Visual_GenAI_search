#!/bin/bash
EMAIL="ellorwaizner.nir@mail.huji.ac.il"
VENV="source /cs/labs/raananf/ellorw.nir/venv/bin/activate"
VENV_JAX="source /cs/labs/raananf/ellorw.nir/venv_jax/bin/activate"
DIMS=(64 128 256 384)

## Step 3 — distill students (PyTorch venv, 1 GPU per dim)
#STEP3_IDS=()
#for DIM in "${DIMS[@]}"; do
#  JOB=$(sbatch \
#    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
#    --mail-type=ALL --mail-user=$EMAIL --job-name=step3_dim${DIM} \
#    --wrap "bash -c '$VENV; python3 step3_distill_students.py --dim $DIM'" \
#    | awk '{print $NF}')
#  echo "Submitted step3 dim=$DIM → Job ID: $JOB"
#  STEP3_IDS+=($JOB)
#done

#STEP3_DEP="afterok:$(IFS=:; echo "${STEP3_IDS[*]}")"

# Step 4a — generate latents (PyTorch venv, 1 GPU per dim)
STEP4A_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch --dependency=$STEP3_DEP \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step4a_dim${DIM} \
    --wrap "bash -c '$VENV; python3 step4_evaluate.py --generate-only $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step4a dim=$DIM → Job ID: $JOB"
  STEP4A_IDS+=($JOB)
done

STEP4A_DEP="afterok:$(IFS=:; echo "${STEP4A_IDS[*]}")"

# Step 4b — decode latents to images (JAX venv, no GPU needed)
STEP4B_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch --dependency=$STEP4A_DEP \
    --mem=30G -c1 --time=13-23 --gres=gpu:0 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step4b_dim${DIM} \
    --wrap "bash -c '$VENV_JAX; python3 step4_evaluate.py --decode-only $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step4b dim=$DIM → Job ID: $JOB"
  STEP4B_IDS+=($JOB)
done

STEP4B_DEP="afterok:$(IFS=:; echo "${STEP4B_IDS[*]}")"

# Step 4c — compute FID/IS (PyTorch venv, 1 GPU per dim)
STEP4C_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch --dependency=$STEP4B_DEP \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step4c_dim${DIM} \
    --wrap "bash -c '$VENV; export TORCH_HOME=/cs/labs/raananf/ellorw.nir/torch_cache; python3 step4_evaluate.py --metrics-only $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step4c dim=$DIM → Job ID: $JOB"
  STEP4C_IDS+=($JOB)
done

STEP4C_DEP="afterok:$(IFS=:; echo "${STEP4C_IDS[*]}")"

# Step 4d — plot (PyTorch venv, no GPU)
JOB_PLOT=$(sbatch --dependency=$STEP4C_DEP \
  --mem=8G -c1 --time=0-01 --gres=gpu:0 \
  --mail-type=ALL --mail-user=$EMAIL --job-name=step4_plot \
  --wrap "bash -c '$VENV; export TORCH_HOME=/cs/labs/raananf/ellorw.nir/torch_cache; python3 step4_evaluate.py --plot-only'" \
  | awk '{print $NF}')
echo "Submitted step4_plot → Job ID: $JOB_PLOT"

echo ""
echo "Pipeline submitted:"
echo "  step3 jobs  : ${STEP3_IDS[*]}"
echo "  step4a jobs : ${STEP4A_IDS[*]}  (wait for all step3)"
echo "  step4b jobs : ${STEP4B_IDS[*]}  (wait for all step4a) — JAX venv"
echo "  step4c jobs : ${STEP4C_IDS[*]}  (wait for all step4b) — PyTorch venv"
echo "  step4_plot  : $JOB_PLOT          (wait for all step4c)"