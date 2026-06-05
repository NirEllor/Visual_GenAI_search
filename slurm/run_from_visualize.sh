#!/bin/bash
# Resume pipeline from existing latents:
# visualize latents → step2 → step3a → step3b → step4 → plot losses

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ids_from() {
  local tag="$1" out="$2"
  grep "^${tag} job IDs:" <<< "$out" | sed "s/^${tag} job IDs: //"
}

ids_to_dep() {
  echo "afterok:$(echo "$1" | tr ' ' ':')"
}

echo "=================================================="
echo "  Pipeline from Visualize Latents"
echo "=================================================="

# ── Visualize Latents ────────────────────────────────────────────────────────
echo ""
echo "=== Visualize Latent Spaces ==="
OUTVIZ=$(bash "$SCRIPT_DIR/run_visualize_latents.sh" pca 5000 64 128 256 384 512 1024)
echo "$OUTVIZ"
IDSVIZ=$(ids_from "visualize_latents" "$OUTVIZ")
DEPVIZ=$(ids_to_dep "$IDSVIZ")

# ── Step 2: Train Teachers ───────────────────────────────────────────────────
echo ""
echo "=== Step 2: Train Teachers ==="
OUT2=$(bash "$SCRIPT_DIR/run_step2.sh" "$DEPVIZ")
echo "$OUT2"
IDS2=$(ids_from "step2" "$OUT2")
DEP2=$(ids_to_dep "$IDS2")

# ── Step 3a: Generate Synthetic Data ─────────────────────────────────────────
echo ""
echo "=== Step 3a: Generate Synthetic Datasets ==="
OUT3A=$(bash "$SCRIPT_DIR/run_step3a.sh" "$DEP2")
echo "$OUT3A"
IDS3A=$(ids_from "step3a" "$OUT3A")
DEP3A=$(ids_to_dep "$IDS3A")

# ── Step 3b: Distil Students ─────────────────────────────────────────────────
echo ""
echo "=== Step 3b: Distil Students ==="
OUT3B=$(bash "$SCRIPT_DIR/run_step3b.sh" "$DEP3A")
echo "$OUT3B"
IDS3B=$(ids_from "step3b" "$OUT3B")
DEP3B=$(ids_to_dep "$IDS3B")

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Evaluate All Models ==="
OUT4=$(bash "$SCRIPT_DIR/run_step4.sh" "$DEP3B")
echo "$OUT4"

# ── Plot Losses ──────────────────────────────────────────────────────────────
echo ""
echo "=== Plot Losses ==="
OUTPL=$(bash "$SCRIPT_DIR/run_plot_losses.sh" "$DEP3B")
echo "$OUTPL"

echo ""
echo "=================================================="
echo "  Submission complete"
echo "=================================================="
echo "  visualize job IDs : $IDSVIZ"
echo "  step2     job IDs : $IDS2"
echo "  step3a    job IDs : $IDS3A"
echo "  step3b    job IDs : $IDS3B"