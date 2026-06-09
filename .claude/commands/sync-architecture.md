# sync-architecture

Sync `architecture.txt` to match the current source in `models/`.

## What to do

1. Read all three model files:
   - `models/autoencoder.py`
   - `models/diffusion.py`
   - `models/denoiser.py`

2. Read the current `architecture.txt`.

3. Compare every section of `architecture.txt` against the source code and fix any
   discrepancy.  The sections to check are:

   **Section 1 — Autoencoders**
   - Encoder conv stack (channels, strides, norms, activations)
   - VAE heads: encoder_mean / encoder_logvar (kernel size, in/out channels)
   - Reparameterization logic and clamp bounds
   - Decoder: decoder_proj, decoder_refine (if present), decoder_conv stack
   - Training hyperparams (loss, optimizer, lr, epochs, batch size, LR schedule,
     grad clip, EMA decay)

   **Section 2 — Flow Matching**
   - Forward process equations and sampling loop (Euler steps, direction)

   **Section 3 — Denoiser Architecture**
   - SinusoidalPosEmb frequency formula and time scaling
   - Time-embedding MLP layout
   - ConvResBlock internals (norms, convs, FiLM projection, forward formula)
   - Output head layers
   - Full forward-pass shape trace
   - TeacherDenoiser: n_blocks default, hidden_channels formula, per-dim table
   - StudentDenoiser: n_blocks default, hidden_channels formula, per-dim table

   **Section 4 — Training Hyperparameters**
   - Teacher: epochs, batch size, optimizer, LR schedule, grad clip, EMA, loss
   - Student: epochs, batch size, optimizer, LR schedule, grad clip, EMA, loss
   - Latent normalisation details

   **Section 5 — Synthetic Data Generation**
   - Sampling steps, batch size
   - Dataset sizes (currently: 50k, 100k, 150k, 200k)
   - Trajectory shape

   **Section 6 — Evaluation**
   - Phases and their outputs
   - Number of generated images per combination

4. Apply only the changes that are actually wrong.  Do not reformat unchanged text.

5. After editing `architecture.txt`, commit and push:
   ```
   git add architecture.txt
   git commit -m "sync architecture.txt with models/"
   git push
   ```
