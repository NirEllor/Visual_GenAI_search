#!/bin/bash
# Step 4 for Config B students (all dims x sizes): generate, decode, metrics, then plot.

set -e

cd /cs/labs/raananf/ellorw.nir/distillation/Distillation_Research
source /cs/labs/raananf/ellorw.nir/venv/bin/activate

EXP="ae_kl_lpips_only_v1_phase12"
DIMS="64 128 256 384 512 1024"
SIZES="50000 100000 150000 200000"

for PHASE in generate decode metrics; do
  echo "════ Step 4 students: --$PHASE ════"
  for DIM in $DIMS; do
    for SIZE in $SIZES; do
      echo "[$(date '+%H:%M:%S')] --$PHASE dim=$DIM size=$SIZE"
      python step4_evaluate.py --$PHASE --dim $DIM --size $SIZE --exp-name $EXP
    done
  done
done

echo "════ Step 4 students: --plot ════"
python step4_evaluate.py --plot --exp-name $EXP

echo "Done. Metrics: results/$EXP/metrics/metrics_all.json"
