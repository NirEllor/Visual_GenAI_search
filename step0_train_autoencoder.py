"""
Step 0 — Train JAX/Flax autoencoders for custom latent dimensions.

Trains autoencoders for latent dims not provided by phlippe's checkpoints.
Uses the same architecture as the UvA Tutorial 9 autoencoder.

Output files
------------
checkpoints/cifar10_{dim}_custom.ckpt   for each dim in CUSTOM_DIMS
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import flax

import torch
import torchvision
import torchvision.transforms as T

# ── configuration ─────────────────────────────────────────────────────────────
CUSTOM_DIMS  = [8, 16, 32]
C_HID        = 32
EPOCHS       = 200
BATCH_SIZE   = 256
LR           = 1e-3
DATA_DIR     = "data"
CKPT_DIR     = "checkpoints"
LOG_INTERVAL = 50


# ── data ──────────────────────────────────────────────────────────────────────

def get_cifar10_numpy():
    import pickle
    import os

    def load_batch(path):
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        return d['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC

    # Download using torchvision (just for download, no transforms)
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    # Load raw batches directly
    cifar_dir = os.path.join(DATA_DIR, 'cifar-10-batches-py')

    train_batches = []
    for i in range(1, 6):
        train_batches.append(load_batch(os.path.join(cifar_dir, f'data_batch_{i}')))
    train_imgs = np.concatenate(train_batches, axis=0).astype(np.float32) / 127.5 - 1.0

    test_imgs = load_batch(os.path.join(cifar_dir, 'test_batch')).astype(np.float32) / 127.5 - 1.0

    return train_imgs, test_imgs


# ── model ──────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    c_hid: int
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.latent_dim)(x)
        return x


class Decoder(nn.Module):
    c_out: int
    c_hid: int
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(2 * 16 * self.c_hid)(x)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)
        x = nn.ConvTranspose(2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.gelu(x)
        x = nn.Conv(self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.ConvTranspose(self.c_out, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.tanh(x)
        return x


class Autoencoder(nn.Module):
    c_hid: int
    latent_dim: int

    def setup(self):
        self.encoder = Encoder(c_hid=self.c_hid, latent_dim=self.latent_dim)
        self.decoder = Decoder(c_hid=self.c_hid, latent_dim=self.latent_dim, c_out=3)

    def __call__(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ── training ───────────────────────────────────────────────────────────────────

def mse_loss(params, model, batch):
    recon = model.apply(params, batch)
    return jnp.mean((recon - batch) ** 2)


@jax.jit
def train_step(state, batch):
    loss, grads = jax.value_and_grad(mse_loss)(state.params, state.apply_fn, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    return mse_loss(state.params, state.apply_fn, batch)


def train_autoencoder(dim: int, train_imgs: np.ndarray, test_imgs: np.ndarray):
    print(f"\n{'='*60}")
    print(f"  Training autoencoder  latent_dim = {dim}")

    out_path = Path(CKPT_DIR) / f"cifar10_{dim}_custom.ckpt"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists.")
        return

    # Init model
    model = Autoencoder(c_hid=C_HID, latent_dim=dim)
    key  = jax.random.PRNGKey(42)
    dummy = jnp.ones((1, 32, 32, 3))
    params = model.init(key, dummy)

    # Optimizer
    tx = optax.adam(LR)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    N = len(train_imgs)
    best_val_loss = float("inf")
    best_params   = params

    for epoch in tqdm(range(1, EPOCHS + 1), desc=f"dim={dim}"):
        # Shuffle
        perm = np.random.permutation(N)
        train_loss = 0.0
        steps = 0

        for start in range(0, N, BATCH_SIZE):
            batch = jnp.array(train_imgs[perm[start:start + BATCH_SIZE]])
            state, loss = train_step(state, batch)
            train_loss += float(loss)
            steps += 1

        train_loss /= steps

        # Validation
        val_loss = 0.0
        val_steps = 0
        for start in range(0, len(test_imgs), BATCH_SIZE):
            batch = jnp.array(test_imgs[start:start + BATCH_SIZE])
            val_loss += float(eval_step(state, batch))
            val_steps += 1
        val_loss /= val_steps

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params   = state.params

        if epoch % LOG_INTERVAL == 0:
            print(f"  epoch {epoch:03d}  train={train_loss:.5f}  val={val_loss:.5f}  best={best_val_loss:.5f}")

    # Save best params
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(flax.serialization.to_bytes(best_params["params"]))
    print(f"  Saved → {out_path}  (best_val_loss={best_val_loss:.5f})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=CUSTOM_DIMS, default=None,
                        help="Single latent dim to train. Omit to train all.")
    args = parser.parse_args()

    dims = [args.dim] if args.dim is not None else CUSTOM_DIMS

    print("Loading CIFAR-10...")
    train_imgs, test_imgs = get_cifar10_numpy()
    print(f"  train: {train_imgs.shape}  test: {test_imgs.shape}")

    for dim in dims:
        train_autoencoder(dim, train_imgs, test_imgs)

    print("\nStep 0 complete.")


if __name__ == "__main__":
    main()