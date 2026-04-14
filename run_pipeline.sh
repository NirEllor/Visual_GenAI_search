#!/bin/bash
EMAIL="ellorwaizner.nir@mail.huji.ac.il"
VENV="source /cs/labs/raananf/ellorw.nir/distillation/Distillation_Research/venv/bin/activate"
DIMS=(64 128 256 384)

# Step 2 — 4 parallel jobs, one per GPU per dim
STEP2_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step2_dim${DIM} \
    --wrap "bash -c '$VENV; python3 step2_train_teachers.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step2 dim=$DIM → Job ID: $JOB"
  STEP2_IDS+=($JOB)
done

STEP2_DEP="afterok:$(IFS=:; echo "${STEP2_IDS[*]}")"

# Step 3 — 4 parallel jobs, one per GPU per dim, waits for all step 2
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

# Step 4 — 4 parallel eval jobs, one per GPU per dim, waits for all step 3
STEP4_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch --dependency=$STEP3_DEP \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=ALL --mail-user=$EMAIL --job-name=step4_dim${DIM} \
    --wrap "bash -c '$VENV; python3 step4_evaluate.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step4 dim=$DIM → Job ID: $JOB"
  STEP4_IDS+=($JOB)
done

STEP4_DEP="afterok:$(IFS=:; echo "${STEP4_IDS[*]}")"

# Step 4 plot — no GPU needed, just merges JSONs and plots, waits for all eval jobs
JOB_PLOT=$(sbatch --dependency=$STEP4_DEP \
  --mem=8G -c1 --time=0-01 --gres=gpu:0 \
  --mail-type=ALL --mail-user=$EMAIL --job-name=step4_plot \
  --wrap "bash -c '$VENV; python3 step4_evaluate.py --plot-only'" \
  | awk '{print $NF}')
echo "Submitted step4_plot → Job ID: $JOB_PLOT"

echo ""
echo "Pipeline submitted:"
echo "  step2 jobs : ${STEP2_IDS[*]}"
echo "  step3 jobs : ${STEP3_IDS[*]}  (wait for all step2)"
echo "  step4 jobs : ${STEP4_IDS[*]}  (wait for all step3)"
echo "  step4_plot : $JOB_PLOT         (wait for all step4)"