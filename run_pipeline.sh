#!/bin/bash
EMAIL="ellorwaizner.nir@mail.huji.ac.il"
VENV="source /cs/labs/raananf/ellorw.nir/distillation/Distillation_Research/venv/bin/activate"
DIMS=(64 128 256 384)

# Step 2 — 4 parallel jobs, one per GPU per dim
STEP2_IDS=()
for DIM in "${DIMS[@]}"; do
  JOB=$(sbatch \
    --mem=30G -c1 --time=13-23 --gres=gpu:1 \
    --mail-type=FAIL --mail-user=$EMAIL --job-name=step2_dim${DIM} \
    --wrap "bash -c '$VENV; python3 step2_train_teachers.py --dim $DIM'" \
    | awk '{print $NF}')
  echo "Submitted step2 dim=$DIM → Job ID: $JOB"
  STEP2_IDS+=($JOB)
done

# Step 3 — waits for ALL 4 step 2 jobs
STEP2_DEP="afterok:$(IFS=:; echo "${STEP2_IDS[*]}")"

JOB3=$(sbatch --dependency=$STEP2_DEP \
  --mem=120G -c1 --time=13-23 --gres=gpu:4 \
  --mail-type=FAIL --mail-user=$EMAIL --job-name=step3_distill \
  --wrap "bash -c '$VENV; python3 step3_distill_students.py'" \
  | awk '{print $NF}')
echo "Submitted step3 → Job ID: $JOB3"

# Step 4 — waits for step 3
JOB4=$(sbatch --dependency=afterok:$JOB3 \
  --mem=120G -c1 --time=13-23 --gres=gpu:4 \
  --mail-type=FAIL --mail-user=$EMAIL --job-name=step4_eval \
  --wrap "bash -c '$VENV; python3 step4_evaluate.py'" \
  | awk '{print $NF}')
echo "Submitted step4 → Job ID: $JOB4"

echo ""
echo "Pipeline submitted:"
echo "  step2 jobs : ${STEP2_IDS[*]}"
echo "  step3      : $JOB3  (waits for all step2)"
echo "  step4      : $JOB4  (waits for step3)"