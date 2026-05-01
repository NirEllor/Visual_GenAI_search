#!/bin/bash
# Step 4 — Evaluate all 30 models (24 students + 6 teachers).
# Internally chains 4 phases: generate → decode → metrics → plot.
# Usage:
#   bash slurm/run_step4.sh                        # no dependency
#   bash slurm/run_step4.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

# ── Phase 1: generate latents (24 students + 6 teachers = 30 jobs) ─────────────
echo "  [step4] Submitting generate phase (30 jobs)..."
GEN_IDS=()

for DIM in "${DIMS[@]}"; do
  for SIZE in "${SIZES[@]}"; do
    JOB=$(sbatch $DEP_FLAG \
      --mem=30G -c2 --time=0-04 --gres=gpu:1 \
      --mail-type=FAIL,END --mail-user="$EMAIL" \
      --job-name=s4_gen_d${DIM}_n${SIZE} \
      --wrap "bash -c '$RUN python /workspace/step4_evaluate.py --generate --dim $DIM --size $SIZE'" \
      | awk '{print $NF}')
    GEN_IDS+=($JOB)
  done

  JOB=$(sbatch $DEP_FLAG \
    --mem=30G -c2 --time=0-04 --gres=gpu:1 \
    --mail-type=FAIL,END --mail-user="$EMAIL" \
    --job-name=s4_gen_tea_d${DIM} \
    --wrap "bash -c '$RUN python /workspace/step4_evaluate.py --generate --teacher --dim $DIM'" \
    | awk '{print $NF}')
  GEN_IDS+=($JOB)
done

echo "  generate job IDs: ${GEN_IDS[*]}"
GEN_DEP="afterok:$(IFS=:; echo "${GEN_IDS[*]}")"

# ── Phase 2: decode latents to images (30 jobs) ─────────────────────────────────
echo "  [step4] Submitting decode phase (30 jobs)..."
DEC_IDS=()

for DIM in "${DIMS[@]}"; do
  for SIZE in "${SIZES[@]}"; do
    JOB=$(sbatch --dependency="$GEN_DEP" \
      --mem=30G -c2 --time=0-04 --gres=gpu:1 \
      --mail-type=FAIL,END --mail-user="$EMAIL" \
      --job-name=s4_dec_d${DIM}_n${SIZE} \
      --wrap "bash -c '$RUN python /workspace/step4_evaluate.py --decode --dim $DIM --size $SIZE'" \
      | awk '{print $NF}')
    DEC_IDS+=($JOB)
  done

  JOB=$(sbatch --dependency="$GEN_DEP" \
    --mem=30G -c2 --time=0-04 --gres=gpu:1 \
    --mail-type=FAIL,END --mail-user="$EMAIL" \
    --job-name=s4_dec_tea_d${DIM} \
    --wrap "bash -c '$RUN python /workspace/step4_evaluate.py --decode --teacher --dim $DIM'" \
    | awk '{print $NF}')
  DEC_IDS+=($JOB)
done

echo "  decode job IDs: ${DEC_IDS[*]}"
DEC_DEP="afterok:$(IFS=:; echo "${DEC_IDS[*]}")"

# ── Phase 3: compute FID + IS metrics (30 jobs) ─────────────────────────────────
echo "  [step4] Submitting metrics phase (30 jobs)..."
MET_IDS=()

for DIM in "${DIMS[@]}"; do
  for SIZE in "${SIZES[@]}"; do
    JOB=$(sbatch --dependency="$DEC_DEP" \
      --mem=40G -c2 --time=0-04 --gres=gpu:1 \
      --mail-type=FAIL,END --mail-user="$EMAIL" \
      --job-name=s4_met_d${DIM}_n${SIZE} \
      --wrap "bash -c '$RUN python /workspace/step4_evaluate.py --metrics --dim $DIM --size $SIZE'" \
      | awk '{print $NF}')
    MET_IDS+=($JOB)
  done

  JOB=$(sbatch --dependency="$DEC_DEP" \
    --mem=40G -c2 --time=0-04 --gres=gpu:1 \
    --mail-type=FAIL,END --mail-user="$EMAIL" \
    --job-name=s4_met_tea_d${DIM} \
    --wrap "bash -c '$RUN python /workspace/step4_evaluate.py --metrics --teacher --dim $DIM'" \
    | awk '{print $NF}')
  MET_IDS+=($JOB)
done

echo "  metrics job IDs: ${MET_IDS[*]}"
MET_DEP="afterok:$(IFS=:; echo "${MET_IDS[*]}")"

# ── Phase 4: aggregate + plot (1 CPU job) ────────────────────────────────────────
JOB_PLOT=$(sbatch --dependency="$MET_DEP" \
  --mem=8G -c1 --time=0-01 --gres=gpu:0 \
  --mail-type=ALL --mail-user="$EMAIL" \
  --job-name=step4_plot \
  --wrap "bash -c '$RUN python /workspace/step4_evaluate.py --plot'" \
  | awk '{print $NF}')
echo "  Submitted step4 plot → Job $JOB_PLOT"

ALL_IDS=("${GEN_IDS[@]}" "${DEC_IDS[@]}" "${MET_IDS[@]}" "$JOB_PLOT")
echo "step4 job IDs: ${ALL_IDS[*]}"
