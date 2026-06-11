#!/bin/bash
# Master pipeline — submits all stages end-to-end with correct dependencies.
# Each stage waits for its predecessor before any of its jobs start.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ids_from() {
  local tag="$1" out="$2"
  grep "^${tag} job IDs:" <<< "$out" | sed "s/^${tag} job IDs: //"
}

ids_to_dep() {
  echo "afterok:$(echo "$1" | tr ' ' ':')"
}

echo "=================================================="
echo "  Full Pipeline Submission"
echo "=================================================="

# ── Step 0: Train AutoEncoders ───────────────────────────────────────────────
echo ""
echo "=== Step 0: Train AutoEncoders ==="
OUT0=$(bash "$SCRIPT_DIR/run_step0.sh")
echo "$OUT0"
IDS0=$(ids_from "step0" "$OUT0")
DEP0=$(ids_to_dep "$IDS0")

# ── Step 0b: Eval AutoEncoders ───────────────────────────────────────────────
echo ""
echo "=== Step 0b: Eval AutoEncoders ==="
OUT0B=$(bash "$SCRIPT_DIR/run_step0b.sh" "$DEP0")
echo "$OUT0B"
IDS0B=$(ids_from "step0b" "$OUT0B")

# ── Step 1: Extract Latents ─────────────────────────────────────────────────
echo ""
echo "=== Step 1: Extract Latents ==="
DEP0_AND_0B="afterok:$(echo "$IDS0 $IDS0B" | tr ' ' ':')"
OUT1=$(bash "$SCRIPT_DIR/run_step1.sh" "$DEP0_AND_0B")
echo "$OUT1"
IDS1=$(ids_from "step1" "$OUT1")
DEP1=$(ids_to_dep "$IDS1")

# ── Step 2: Train Teachers ──────────────────────────────────────────────────
echo ""
echo "=== Step 2: Train Teachers ==="
OUT2=$(bash "$SCRIPT_DIR/run_step2.sh" "$DEP1")
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
echo "  step0      job IDs : $IDS0"
echo "  step0b     job IDs : $IDS0B"
echo "  step1      job IDs : $IDS1"
echo "  step2      job IDs : $IDS2"
echo "  step3a     job IDs : $IDS3A"
echo "  step3b     job IDs : $IDS3B"
echo "  step4      job IDs : $IDS4"
echo "  visualize  job IDs : $IDSVIZ"
echo "  plot_losses depends on step3b — see output above for its IDs"