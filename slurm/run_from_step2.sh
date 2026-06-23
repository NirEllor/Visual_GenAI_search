#!/bin/bash
# Pipeline starting from Step 2 (Train Teachers) — assumes AEs and latents are ready.
# Submits Steps 2 → 3a → 3b → 4 → plot_losses → visualize with correct dependencies.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ids_from() {
  local tag="$1" out="$2"
  grep "^${tag} job IDs:" <<< "$out" | sed "s/^${tag} job IDs: //"
}

ids_to_dep() {
  echo "afterok:$(echo "$1" | tr ' ' ':')"
}

echo "=================================================="
echo "  Pipeline from Step 2 (Train Teachers)"
echo "=================================================="

# ── Step 2: Train Teachers ──────────────────────────────────────────────────
echo ""
echo "=== Step 2: Train Teachers ==="
OUT2=$(bash "$SCRIPT_DIR/run_step2.sh")
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

# ── Step 3b: Distil Students ────────────────────────────────────────────────
echo ""
echo "=== Step 3b: Distil 24 Students ==="
OUT3B=$(bash "$SCRIPT_DIR/run_step3b.sh" "$DEP3A")
echo "$OUT3B"
IDS3B=$(ids_from "step3b" "$OUT3B")
DEP3B=$(ids_to_dep "$IDS3B")

# ── Step 4: Evaluate ────────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Evaluate All Models ==="
OUT4=$(bash "$SCRIPT_DIR/run_step4.sh" "$DEP3B")
echo "$OUT4"
IDS4=$(ids_from "step4" "$OUT4")
DEP4=$(ids_to_dep "$IDS4")

# ── Plot Losses ─────────────────────────────────────────────────────────────
echo ""
echo "=== Plot Losses ==="
OUTPL=$(bash "$SCRIPT_DIR/run_plot_losses.sh" "$DEP3B")
echo "$OUTPL"

# ── Visualize Latents — final stage, after Step 4 ────────────────────────────
echo ""
echo "=== Visualize Latent Spaces ==="
OUTVIZ=$(bash "$SCRIPT_DIR/run_visualize_latents.sh" all 3000 64 128 256 384 512 1024 "$DEP4")
echo "$OUTVIZ"
IDSVIZ=$(ids_from "visualize_latents" "$OUTVIZ")

echo ""
echo "=================================================="
echo "  Submission complete"
echo "=================================================="
echo "  step2      job IDs : $IDS2"
echo "  step3a     job IDs : $IDS3A"
echo "  step3b     job IDs : $IDS3B"
echo "  step4      job IDs : $IDS4"
echo "  visualize  job IDs : $IDSVIZ"
echo "  plot_losses depends on step3b — see output above for its IDs"
