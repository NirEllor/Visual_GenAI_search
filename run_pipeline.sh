#!/bin/bash
EMAIL="ellorwaizner.nir@mail.huji.ac.il"
VENV="source /cs/labs/raananf/ellorw.nir/venv/bin/activate"
VENV_JAX="source /cs/labs/raananf/ellorw.nir/venv_jax/bin/activate"
DIMS=(8 16 32 64 128 256 384)

# Step 0 — train custom autoencoders (JAX venv, 1 GPU per dim, only for new dims)
STEP0_IDS=()
for DIM in 8 16 32; do
  JOB=$(sbatch \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step0_dim${DIM} \
    --wrap "bash -c '$VENV_JAX; python3 step0_train_autoencoder.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step0 dim=$DIM → Job ID: $JOB"
  STEP0_IDS+=($JOB)
done

STEP0_DEP="afterok:$(IFS=:; echo "${STEP0_IDS[*]}")"

# Step 1 — extract latents (JAX venv, 1 GPU per dim)
# Only new dims need step0 dependency; existing dims can start immediately
STEP1_IDS=()
for DIM in "${DIMS[@]}"; do
  if [[ "$DIM" == "8" || "$DIM" == "16" || "$DIM" == "32" ]]; then
    DEP="--dependency=$STEP0_DEP"
  else
    DEP=""
  fi
  JOB=$(sbatch $DEP \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step1_dim${DIM} \
    --wrap "bash -c '$VENV_JAX; python3 step1_extract_latents.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step1 dim=$DIM → Job ID: $JOB"
  STEP1_IDS+=($JOB)
done

STEP1_DEP="afterok:$(IFS=:; echo "${STEP1_IDS[*]}")"

# Step 2 — train teachers (PyTorch venv, 1 GPU per dim)
STEP2_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch --dependency=$STEP1_DEP \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step2_dim${DIM} \
    --wrap "bash -c '$VENV; python3 step2_train_teachers.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step2 dim=$DIM → Job ID: $JOB"
  STEP2_IDS+=($JOB)
done

STEP2_DEP="afterok:$(IFS=:; echo "${STEP2_IDS[*]}")"

# Step 3 — distill students (PyTorch venv, 1 GPU per dim)
STEP3_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch --dependency=$STEP2_DEP \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step3_dim${DIM} \
    --wrap "bash -c '$VENV; python3 step3_distill_students.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step3 dim=$DIM → Job ID: $JOB"
  STEP3_IDS+=($JOB)
done

STEP3_DEP="afterok:$(IFS=:; echo "${STEP3_IDS[*]}")"

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
echo "  step0 jobs  : ${STEP0_IDS[*]}  (new dims only)"
echo "  step1 jobs  : ${STEP1_IDS[*]}  (8,16,32 wait for step0)"
echo "  step2 jobs  : ${STEP2_IDS[*]}  (wait for all step1)"
echo "  step3 jobs  : ${STEP3_IDS[*]}  (wait for all step2)"
echo "  step4a jobs : ${STEP4A_IDS[*]}  (wait for all step3)"
echo "  step4b jobs : ${STEP4B_IDS[*]}  (wait for all step4a) — JAX venv"
echo "  step4c jobs : ${STEP4C_IDS[*]}  (wait for all step4b) — PyTorch venv"
echo "  step4_plot  : $JOB_PLOT          (wait for all step4c)"