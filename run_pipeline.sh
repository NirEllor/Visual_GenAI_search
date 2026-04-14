VENV="source /cs/labs/raananf/ellorw.nir/venv/bin/activate"         # PyTorch + NumPy<2
VENV_JAX="source /cs/labs/raananf/ellorw.nir/venv_jax/bin/activate" # JAX + NumPy>=2

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

# Step 4b — decode + FID (JAX venv, CPU is fine)
STEP4B_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch --dependency=$STEP4A_DEP \
    --mem=30G -c1 --time=13-23 --gres=gpu:0 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step4b_dim${DIM} \
    --wrap "bash -c '$VENV_JAX; python3 step4_evaluate.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step4b dim=$DIM → Job ID: $JOB"
  STEP4B_IDS+=($JOB)
done

STEP4B_DEP="afterok:$(IFS=:; echo "${STEP4B_IDS[*]}")"

# Step 4c — plot (no GPU)
JOB_PLOT=$(sbatch --dependency=$STEP4B_DEP \
  --mem=8G -c1 --time=0-01 --gres=gpu:0 \
  --mail-type=ALL --mail-user=$EMAIL --job-name=step4_plot \
  --wrap "bash -c '$VENV_JAX; python3 step4_evaluate.py --plot-only'" \
  | awk '{print $NF}')
echo "Submitted step4_plot → Job ID: $JOB_PLOT"