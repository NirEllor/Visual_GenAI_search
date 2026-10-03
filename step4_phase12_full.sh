#!/bin/bash
# Step 4 Phase 1+2 - Full evaluation (generate, decode, metrics, plot)

set -e  # Exit on error

cd /cs/labs/raananf/ellorw.nir/distillation/Distillation_Research
source /cs/labs/raananf/ellorw.nir/venv/bin/activate

EXP="ae_kl_lpips_only_v1_phase12"

echo "════════════════════════════════════════════════════════════"
echo "Step 4 Phase 1+2: Full Evaluation"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "Step 4a: Generate latent samples for all dims"
echo "────────────────────────────────────────────────────────────"
for DIM in 64 128 256 384 512 1024; do
  echo "[$(date '+%H:%M:%S')] Generating dim=$DIM..."
  python step4_evaluate.py --generate --dim $DIM --teacher --exp-name $EXP
done

echo ""
echo "Step 4b: Decode latents to images for all dims"
echo "────────────────────────────────────────────────────────────"
for DIM in 64 128 256 384 512 1024; do
  echo "[$(date '+%H:%M:%S')] Decoding dim=$DIM..."
  python step4_evaluate.py --decode --dim $DIM --teacher --exp-name $EXP
done

echo ""
echo "Step 4c: Compute metrics for all dims"
echo "────────────────────────────────────────────────────────────"
for DIM in 64 128 256 384 512 1024; do
  echo "[$(date '+%H:%M:%S')] Metrics dim=$DIM..."
  python step4_evaluate.py --metrics --dim $DIM --teacher --exp-name $EXP
done

echo ""
echo "Step 4d: Generate final plots"
echo "────────────────────────────────────────────────────────────"
python step4_evaluate.py --plot --exp-name $EXP

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Step 4 Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Metrics saved to:"
echo "  results/$EXP/metrics/metrics_all.json"
echo ""
echo "Plots saved to:"
echo "  results/$EXP/plots/"
echo ""
