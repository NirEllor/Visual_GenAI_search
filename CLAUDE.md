# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a research project on knowledge distillation, currently in early setup. The virtual environment is in `.venv/`.

## Repository

- GitHub: https://github.com/NirEllor/Distillation_Research
- Clone: `git clone https://github.com/NirEllor/Distillation_Research.git`

## Committing Changes

After every task, commit and push all changes:

```bash
git add .
git commit -m "<clear description of what was done>"
git push
```

## Environment

- Python virtual environment: `.venv/`
- Activate: `source .venv/Scripts/activate` (Windows/bash)
- Install dependencies: `pip install -r requirements.txt` (once created)
- Run scripts: `python <script>.py`

---

## Research Plan: Knowledge Distillation Pipeline for Latent Diffusion on CIFAR-10

**Goal:** Study how latent space dimensionality (64, 128, 256, 384) affects the FID score of distilled diffusion models.

### File Structure

```
Guided_Research/
├── requirements.txt
├── checkpoints/              # Downloaded .ckpt files (gitignored)
├── data/                     # Auto-downloaded CIFAR-10
├── latents/                  # latents_64.npy, latents_128.npy, ...
├── models/
│   ├── autoencoder_jax.py    # JAX/Flax AE architecture (matches Tutorial 9)
│   ├── diffusion.py          # DDPM noise schedule + sampling utilities
│   └── denoiser.py           # MLP denoiser (teacher & student variants)
├── step1_extract_latents.py
├── step2_train_teachers.py
├── step3_distill_students.py
├── step4_evaluate.py
└── results/                  # Saved metrics, plots, generated images
```

### Pipeline Steps

**Step 1 — `step1_extract_latents.py`**
- Load 4 pretrained JAX/Flax autoencoders from UvA Tutorial 9 (dims 64/128/256/384)
- Checkpoints: `https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial9/cifar10_{dim}.ckpt`
- Encode all 50k CIFAR-10 training images through each frozen AE encoder
- Save: `latents/latents_{64,128,256,384}.npy` — shape `(50000, dim)`

**Step 2 — `step2_train_teachers.py`**
- For each latent dim, train a DDPM teacher MLP denoiser on the saved latents
- Architecture: 4 residual blocks, hidden_dim=512, sinusoidal time embedding, T=1000 steps (cosine schedule)
- Train 200 epochs, AdamW lr=3e-4, batch_size=256
- Save: `models/teacher_{64,128,256,384}.pt`

**Step 3 — `step3_distill_students.py`**
- Distill each teacher into a smaller student via **progressive distillation** (Salimans & Ho 2022)
- 1000→500→...→4 steps across 3 rounds (100 epochs/round, lr=1e-4)
- Student architecture: 2 residual blocks, hidden_dim=256 (~4x smaller than teacher)
- Loss: MSE between student and teacher x₀-predictions in distilled step space
- Save: `models/student_{64,128,256,384}.pt`

**Step 4 — `step4_evaluate.py`**
- Generate 10,000 images per student (4-step sampling → JAX AE decode → pixels)
- Compute FID (clean-fid), IS (torch-fidelity), optional LPIPS
- Plot FID vs latent dim → `results/fid_vs_dim.png`
- Save all metrics → `results/metrics.json`

### Key Design Decisions

- Autoencoders are **frozen** throughout — used only for encode/decode
- JAX/Flax used only in Steps 1 & 4; all training is in **PyTorch**
- Bridge JAX↔PyTorch via numpy: `np.array(jax_tensor)` → `torch.from_numpy()`
- Normalize latents to zero-mean unit-variance before diffusion training
- Checkpoint loading: try `orbax.checkpoint` first, fallback to `flax.training.checkpoints`

### Execution Order

```bash
pip install -r requirements.txt
python step1_extract_latents.py      # ~30 min (CPU JAX encoding)
python step2_train_teachers.py       # ~2–4 hrs (GPU)
python step3_distill_students.py     # ~1–2 hrs (GPU)
python step4_evaluate.py             # ~30 min
```

### Pitfalls

| Risk | Mitigation |
|------|-----------|
| JAX AE architecture mismatch | Inspect checkpoint keys with `jax.tree_util.tree_map` |
| Latent scale mismatch | Normalize latents before training |
| Progressive distillation instability | Use EMA of teacher; clip gradients |
| Flax checkpoint format varies | Try orbax first, fallback to legacy |