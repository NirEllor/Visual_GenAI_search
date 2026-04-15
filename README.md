# Latent Dimensionality & Knowledge Distillation in Diffusion Models

A research pipeline investigating how **latent space dimensionality** affects the image quality of distilled diffusion models, measured by FID score on CIFAR-10.

---

## Overview

This project trains DDPM teacher models in latent spaces of varying dimensionality (64, 128, 256, 384), distills each into a compact student model, and evaluates the generated image quality. The central question: *does a higher-dimensional latent space lead to better or worse FID after distillation?*

### Method Summary

```
CIFAR-10 images
      │
      ▼
 Frozen JAX/Flax Autoencoder  (4 variants: dim ∈ {64, 128, 256, 384})
      │  encode
      ▼
 Normalized Latents  (zero-mean, unit-variance)
      │
      ▼
 Teacher DDPM (MLP, 4 res-blocks, hidden=512, T=1000)
      │  knowledge distillation
      ▼
 Student DDPM (MLP, 2 res-blocks, hidden=256)
      │  DDIM, 4 steps
      ▼
 Decoded Images  →  FID / IS / LPIPS
```

---

## Repository Structure

```
Guided_Research/
├── requirements.txt
├── models/
│   ├── autoencoder_jax.py      # JAX/Flax AE architecture (UvA Tutorial 9)
│   ├── diffusion.py            # DDPM cosine noise schedule + sampling
│   └── denoiser.py             # TeacherDenoiser & StudentDenoiser (PyTorch MLP)
├── step1_extract_latents.py    # Encode CIFAR-10 → latents/*.npy
├── step2_train_teachers.py     # Train teacher denoisers
├── step3_distill_students.py   # Distil teachers into students
├── step4_evaluate.py           # Generate images, compute metrics, plot results
├── checkpoints/                # Downloaded AE checkpoints (gitignored)
├── latents/                    # Encoded + normalised latents (gitignored)
├── models_saved/               # Trained .pt checkpoints (gitignored)
└── results/                    # metrics.json, fid_vs_dim.png
```

---

## Installation

```bash
git clone https://github.com/NirEllor/Distillation_Research.git
cd Distillation_Research

python -m venv .venv
source .venv/Scripts/activate   # Windows/bash
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

> **Note:** JAX is installed in CPU-only mode (used only for autoencoder encode/decode). All diffusion training runs in PyTorch on GPU.

---

## Pipeline

### Step 1 — Extract Latents

Downloads 4 pretrained JAX/Flax autoencoders from [UvA Tutorial 9](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html) and encodes all 50,000 CIFAR-10 training images.

```bash
python step1_extract_latents.py
```

**Output:** `latents/latents_{64,128,256,384}.npy` — shape `(50000, dim)`

---

### Step 2 — Train Teacher Models

For each latent dimension, normalises the latents and trains a DDPM teacher MLP denoiser.

| Hyperparameter | Value |
|---|---|
| Architecture | MLP, 4 residual blocks, hidden\_dim=512 |
| Time embedding | Sinusoidal, dim=256 |
| Noise schedule | Cosine, T=1000 |
| Epochs | 200 |
| Optimiser | AdamW, lr=3e-4, cosine decay |
| Batch size | 256 |

```bash
python step2_train_teachers.py
```

**Output:** `models/teacher_{64,128,256,384}.pt`

---

### Step 3 — Distil Student Models

Distils each teacher into a ~4x smaller student using a combined loss:

$$\mathcal{L} = 0.5 \cdot \text{MSE}(\hat{\varepsilon}_\text{student},\, \varepsilon_\text{true}) \;+\; 0.5 \cdot \text{MSE}(\hat{x}_{0,\text{student}},\, \hat{x}_{0,\text{teacher}})$$

| Hyperparameter | Value |
|---|---|
| Architecture | MLP, 2 residual blocks, hidden\_dim=256 |
| Epochs | 150 |
| Optimiser | AdamW, lr=1e-4, cosine decay |
| Inference | DDIM, 4 steps |

```bash
python step3_distill_students.py
```

**Output:** `models/student_{64,128,256,384}.pt`

---

### Step 4 — Evaluate

Generates 10,000 images per student model, decodes them through the frozen autoencoder, and computes image quality metrics.

```bash
python step4_evaluate.py
```

**Metrics computed:** FID (clean-fid), Inception Score (torch-fidelity), LPIPS  
**Output:** `results/metrics.json`, `results/fid_vs_dim.png`

---

## Parallel Execution (Multi-GPU)

Steps 2–4 accept a `--dim` flag to pin a single latent dimension to a specific GPU (`64→cuda:0`, `128→cuda:1`, `256→cuda:2`, `384→cuda:3`).

```bash
# Step 2 — train all 4 teachers in parallel
python step2_train_teachers.py --dim 64  &
python step2_train_teachers.py --dim 128 &
python step2_train_teachers.py --dim 256 &
python step2_train_teachers.py --dim 384 &
wait

# Step 3 — distil all 4 students in parallel
python step3_distill_students.py --dim 64  &
python step3_distill_students.py --dim 128 &
python step3_distill_students.py --dim 256 &
python step3_distill_students.py --dim 384 &
wait

# Step 4 — evaluate in parallel, then merge and plot
python step4_evaluate.py --dim 64  &
python step4_evaluate.py --dim 128 &
python step4_evaluate.py --dim 256 &
python step4_evaluate.py --dim 384 &
wait
python step4_evaluate.py --plot-only
```

Each `--dim` run writes `results/metrics_{dim}.json`. `--plot-only` merges them and produces the final plot without requiring a GPU.

Omit `--dim` to fall back to sequential single-device execution.

---

## Architecture Details

### TeacherDenoiser

```
x_t (B, latent_dim)  →  Linear(hidden=512)  →  4 × ResBlock  →  LayerNorm  →  Linear(latent_dim)
t   (B,)             →  SinusoidalPosEmb    →  MLP           →  injected per ResBlock
```

### StudentDenoiser

Same architecture with `hidden_dim=256` and 2 residual blocks (~4x fewer parameters).

### ResBlock

```
x  →  LayerNorm  →  Linear  →  GELU  →  + time_proj(t_emb)  →  Linear  →  + x
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Autoencoders frozen throughout | Isolates the effect of latent dimensionality on the diffusion model |
| JAX only in Steps 1 & 4 | Reuses pretrained UvA Tutorial 9 checkpoints without reimplementing the AE |
| JAX ↔ PyTorch bridge via NumPy | `np.array(jax_tensor)` → `torch.from_numpy()` — zero-copy on CPU |
| Latent normalisation | Zero-mean, unit-variance prevents scale mismatch across dimensions |
| Student uses DDIM (4 steps) | Drastically reduces inference cost while preserving quality |
| Combined distillation loss | MSE on both noise prediction and x₀ prediction improves student stability |

---

## Runtime Estimates

| Step | Hardware | Estimated Time |
|---|---|---|
| Step 1 — Extract latents | CPU (JAX) | ~30 min |
| Step 2 — Train teachers | GPU (PyTorch) | ~2–4 hrs (sequential) |
| Step 3 — Distil students | GPU (PyTorch) | ~1–2 hrs (sequential) |
| Step 4 — Evaluate | GPU + CPU | ~30 min |

---

## Dependencies

| Package | Purpose |
|---|---|
| `jax`, `flax`, `optax`, `orbax-checkpoint`, `msgpack` | Autoencoder loading & inference |
| `torch`, `torchvision` | Diffusion model training & distillation |
| `clean-fid` | FID computation |
| `torch_fidelity` | Inception Score |
| `lpips` | Perceptual similarity metric |
| `numpy`, `tqdm`, `matplotlib`, `scipy`, `Pillow` | Utilities & visualisation |

---

## Pretrained Autoencoder Checkpoints

Checkpoints are downloaded automatically in Step 1 from:

```
https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial9/cifar10_{dim}.ckpt
```

Credit: [Philip Lippe — UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/)

---

## Productionisation Roadmap

While this repository focuses on research, the pipeline is designed to evolve into a fully deployed product. The planned production stack is outlined below.

### Backend API

A REST/gRPC backend will wrap the inference pipeline and expose it as a service:

- **Framework:** FastAPI (Python) — serves generation requests, returns image URLs or base64-encoded outputs
- **Inference endpoint:** accepts a latent dimension and optional prompt/seed, runs the 4-step DDIM student model, decodes via the frozen autoencoder, and returns the generated image
- **Model serving:** models loaded once at startup and kept in GPU memory; ONNX export or TorchScript compilation for lower latency
- **Request queue:** Celery + Redis for async job handling when GPU throughput is limited

### Deployment

- **Containerisation:** Docker image bundling the FastAPI app, model checkpoints, and CUDA runtime
- **Orchestration:** Kubernetes (EKS on AWS) for auto-scaling inference pods based on request load
- **CI/CD:** GitHub Actions pipeline — linting, unit tests, Docker build, and push to ECR on every merge to `main`

### AWS Infrastructure

| Service | Role |
|---|---|
| **EC2 / EKS** | GPU inference nodes (g4dn or g5 instances) |
| **S3** | Model checkpoint storage, generated image storage |
| **ECR** | Docker image registry |
| **CloudFront** | CDN for serving generated images at low latency |
| **SQS** | Decoupled job queue for batch generation requests |
| **CloudWatch** | Metrics, logging, and alerting for inference latency and error rates |
| **IAM** | Least-privilege roles for each service component |

### Monitoring & Observability

- Prometheus + Grafana dashboards for GPU utilisation, inference latency (p50/p95/p99), and throughput
- Structured JSON logging shipped to CloudWatch Logs
- Alerts on latency regressions or error rate spikes

### Frontend (Planned)

A lightweight web UI allowing users to select a latent dimension, trigger generation, and compare outputs side-by-side across model variants — enabling non-technical stakeholders to explore the research results interactively.

---

## Pretrained Autoencoder Checkpoints

Checkpoints are downloaded automatically in Step 1 from:

```
https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial9/cifar10_{dim}.ckpt
```

Credit: [Philip Lippe — UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/)

---

## License

This project is for academic research purposes.
