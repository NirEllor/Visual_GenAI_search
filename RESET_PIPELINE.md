# Pipeline Reset Commands

Run these from the project root on the cluster.
Each section removes exactly the outputs of that step and everything downstream.
Only go as deep as needed — there is no reason to delete earlier checkpoints if they are still valid.

---

## Level 1 — Retrain the AE (from step 0)

Invalidates the entire pipeline. Remove everything.

```bash
# Step 0: AE checkpoints
rm -f checkpoints/ae_*.pt

# Step 0b: AE eval plots (from step0b_eval_ae.py)
rm -rf results/ae_eval/

# Step 1: Extracted latents
rm -f latents/latents_*.npy

# Step 2: Teachers and normalisation stats
rm -f models/teacher_*.pt
rm -f results/trained_AE/teacher_loss_*.png

# Step 3a: Synthetic datasets and trajectories
rm -rf synthetic/

# Step 3b: Students and their loss plots
rm -f models/student_*.pt
rm -f results/trained_AE/student_loss_*.png

# Step 4: All evaluation outputs
rm -f results/trained_AE/z_orig_*.npy
rm -rf results/trained_AE/generated_*/
rm -rf results/trained_AE/ae_reconstructed_*/
rm -f results/trained_AE/metrics_*.json
rm -f results/trained_AE/metrics_all.json
rm -f results/trained_AE/fid_vs_dim.png
rm -f results/trained_AE/fid_vs_size.png

# Utility plots (plot_losses.py) — optional
rm -f results/trained_AE/losses_*.png
```

---

## Level 2 — Retrain the teachers (from step 2)

Keeps: AE checkpoints (`checkpoints/ae_*.pt`) and raw extracted latents (`latents/latents_{dim}.npy`).

```bash
# Step 2: Norm stats and teacher checkpoints
rm -f latents/latents_*_norm_stats.npy
rm -f models/teacher_*.pt
rm -f results/trained_AE/teacher_loss_*.png

# Step 3a: Synthetic datasets and trajectories
rm -rf synthetic/

# Step 3b: Students and their loss plots
rm -f models/student_*.pt
rm -f results/trained_AE/student_loss_*.png

# Step 4: All evaluation outputs
rm -f results/trained_AE/z_orig_*.npy
rm -rf results/trained_AE/generated_*/
rm -rf results/trained_AE/ae_reconstructed_*/
rm -f results/trained_AE/metrics_*.json
rm -f results/trained_AE/metrics_all.json
rm -f results/trained_AE/fid_vs_dim.png
rm -f results/trained_AE/fid_vs_size.png

# Utility plots (plot_losses.py) — optional
rm -f results/trained_AE/losses_*.png
```

---

## Level 3 — Retrain the students (from step 3b)

Keeps: AE, latents, norm stats, teachers, and all synthetic datasets.

```bash
# Step 3b: Student checkpoints and loss plots
rm -f models/student_*.pt
rm -f results/trained_AE/student_loss_*.png

# Step 4 — student outputs only (teacher z_orig / generated / metrics are kept)
rm -f results/trained_AE/z_orig_*_[0-9]*.npy
rm -rf results/trained_AE/generated_*_[0-9]*/

# ae_reconstructed_{dim}/ depends only on the AE — keep it

# Student metrics + aggregated plots
rm -f results/trained_AE/metrics_*_[0-9]*.json
rm -f results/trained_AE/metrics_all.json
rm -f results/trained_AE/fid_vs_dim.png
rm -f results/trained_AE/fid_vs_size.png
```

---

## Level 4 — Re-evaluate everything (from step 4)

Keeps: AE, latents, teachers, students, synthetic datasets.

### 4a — Re-evaluate both teachers and students

```bash
rm -f results/trained_AE/z_orig_*.npy
rm -rf results/trained_AE/generated_*/
rm -rf results/trained_AE/ae_reconstructed_*/
rm -f results/trained_AE/metrics_*.json
rm -f results/trained_AE/metrics_all.json
rm -f results/trained_AE/fid_vs_dim.png
rm -f results/trained_AE/fid_vs_size.png
```

### 4b — Re-evaluate students only (keep teacher eval results)

```bash
rm -f results/trained_AE/z_orig_*_[0-9]*.npy
rm -rf results/trained_AE/generated_*_[0-9]*/
rm -f results/trained_AE/metrics_*_[0-9]*.json
rm -f results/trained_AE/metrics_all.json
rm -f results/trained_AE/fid_vs_dim.png
rm -f results/trained_AE/fid_vs_size.png
```

### 4c — Re-plot only (metrics JSON already computed, just regenerate the figures)

```bash
rm -f results/trained_AE/metrics_all.json
rm -f results/trained_AE/fid_vs_dim.png
rm -f results/trained_AE/fid_vs_size.png
```

---

## File ownership per step

| File / Directory | Produced by |
|---|---|
| `checkpoints/ae_{dim}.pt` | step 0 |
| `results/ae_eval/` | step 0b |
| `results/trained_AE/losses_{dim}.png` | `plot_losses.py` (utility) |
| `latents/latents_{dim}.npy` | step 1 |
| `latents/latents_{dim}_norm_stats.npy` | step 2 |
| `models/teacher_{dim}.pt` | step 2 |
| `models/teacher_{dim}_ep{N}.pt` | step 2 (interim, every 50 epochs) |
| `results/trained_AE/teacher_loss_{dim}.png` | step 2 |
| `synthetic/{dim}/synthetic_{dim}_{n}.npy` | step 3a |
| `synthetic/{dim}/trajectories_{dim}.npy` | step 3a |
| `models/student_{dim}_{n}.pt` | step 3b |
| `results/trained_AE/student_loss_{dim}_{n}.png` | step 3b |
| `results/trained_AE/z_orig_{dim}_{tag}.npy` | step 4 `--generate` |
| `results/trained_AE/generated_{dim}_{tag}/` | step 4 `--decode` |
| `results/trained_AE/ae_reconstructed_{dim}/` | step 4 `--decode` (once per dim, AE only) |
| `results/trained_AE/metrics_{dim}_{tag}.json` | step 4 `--metrics` |
| `results/trained_AE/metrics_all.json` | step 4 `--plot` |
| `results/trained_AE/fid_vs_dim.png` | step 4 `--plot` |
| `results/trained_AE/fid_vs_size.png` | step 4 `--plot` |

`{tag}` is either `teacher` or the dataset size (e.g. `50000`, `200000`).
`[0-9]*` in the globs matches the numeric size tags only, leaving `_teacher` files untouched.
