#!/bin/bash
# Plot training loss curves — 6 figures, one per latent dim.
# Each figure: teacher loss (left) + 4 student losses (right).
# Output: results/trained_AE/losses_{dim}.png
# Usage:
#   bash slurm/run_plot_losses.sh                        # no dependency
#   bash slurm/run_plot_losses.sh afterok:JID1:JID2:...  # with prior dependency

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$(dirname "$SCRIPT_DIR")"

DEP_FLAG=""
[ -n "${1:-}" ] && DEP_FLAG="--dependency=$1"

JOB=$(sbatch $DEP_FLAG \
  --mem=8G -c1 --time=0-01 --gres=gpu:0 \
  --mail-type=ALL --mail-user="$EMAIL" \
  --job-name=plot_losses \
  --wrap "bash -c '$VENV && python3 plot_losses.py'" \
  | awk '{print $NF}')
echo "  Submitted plot_losses → Job $JOB"
echo "plot_losses job IDs: $JOB"
