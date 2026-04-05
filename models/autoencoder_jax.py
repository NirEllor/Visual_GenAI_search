"""
JAX/Flax autoencoder matching the UvA Deep Learning Tutorial 9 architecture.
Used exclusively for encoding CIFAR-10 images to latents (step 1) and
decoding latents back to images (step 4). Weights are always frozen.

Architecture reference:
  https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.ipynb

Checkpoint source:
  https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial9/
"""

import os
import urllib.request
from urllib.error import HTTPError
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Architecture ──────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    c_hid: int
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # x: (B, 32, 32, 3)  values in [-1, 1]
        x = nn.Conv(self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)   # (B,16,16,c_hid)
        x = nn.gelu(x)
        x = nn.Conv(self.c_hid, kernel_size=(3, 3))(x)                    # (B,16,16,c_hid)
        x = nn.gelu(x)
        x = nn.Conv(2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)  # (B,8,8,2*c_hid)
        x = nn.gelu(x)
        x = nn.Conv(2 * self.c_hid, kernel_size=(3, 3))(x)               # (B,8,8,2*c_hid)
        x = nn.gelu(x)
        x = nn.Conv(2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)  # (B,4,4,2*c_hid)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)                                     # (B, 4*4*2*c_hid)
        x = nn.Dense(self.latent_dim)(x)                                  # (B, latent_dim)
        return x


class Decoder(nn.Module):
    c_out: int
    c_hid: int
    latent_dim: int  # kept for checkpoint key compatibility, unused in forward pass

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(2 * 16 * self.c_hid)(x)                             # (B, 32*c_hid)
        x = nn.gelu(x)
        x = x.reshape(x.shape[0], 4, 4, -1)                              # (B, 4, 4, 2*c_hid)
        x = nn.ConvTranspose(2 * self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)  # (B,8,8,2*c_hid)
        x = nn.gelu(x)
        x = nn.Conv(2 * self.c_hid, kernel_size=(3, 3))(x)               # (B,8,8,2*c_hid)
        x = nn.gelu(x)
        x = nn.ConvTranspose(self.c_hid, kernel_size=(3, 3), strides=(2, 2))(x)      # (B,16,16,c_hid)
        x = nn.gelu(x)
        x = nn.Conv(self.c_hid, kernel_size=(3, 3))(x)                   # (B,16,16,c_hid)
        x = nn.gelu(x)
        x = nn.ConvTranspose(self.c_out, kernel_size=(3, 3), strides=(2, 2))(x)      # (B,32,32,c_out)
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


# ── Checkpoint utilities ──────────────────────────────────────────────────────

CHECKPOINT_BASE_URL = (
    "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial9/"
)
CHECKPOINT_NAMES = {
    64:  "cifar10_64.ckpt",
    128: "cifar10_128.ckpt",
    256: "cifar10_256.ckpt",
    384: "cifar10_384.ckpt",
}
C_HID = 32  # base channel width used in all Tutorial 9 checkpoints


def download_checkpoints(save_dir: str = "checkpoints") -> None:
    """Download all four autoencoder checkpoints if not already present."""
    os.makedirs(save_dir, exist_ok=True)

    for file_name in CHECKPOINT_NAMES.values():
        file_path = os.path.join(save_dir, file_name)
        if not os.path.isfile(file_path):
            file_url = CHECKPOINT_BASE_URL + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please contact the author with the full output including the following error:\n", e)


def _try_load_bytes(path: Path, variables):
    """Try to deserialize a raw Flax msgpack checkpoint file."""
    import flax
    with open(path, "rb") as f:
        data = f.read()
    return flax.serialization.from_bytes(variables, data)


def _try_load_orbax(path: Path):
    """Try to load an Orbax checkpoint (directory or single-file)."""
    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(str(path))


def _try_load_legacy(path: Path, variables):
    """Try legacy flax.training.checkpoints directory format."""
    from flax.training import checkpoints
    # restore_checkpoint expects a directory; if path is a file, use its parent
    ckpt_dir = str(path.parent) if path.is_file() else str(path)
    return checkpoints.restore_checkpoint(ckpt_dir, target=variables)


def load_autoencoder(ckpt_path: str, latent_dim: int, c_hid: int = C_HID):
    """
    Load a pretrained autoencoder from a checkpoint file.

    Returns
    -------
    model : Autoencoder  (Flax module, call model.apply(params, x) )
    params : dict        ({'params': ...})
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run download_checkpoints() first."
        )

    model = Autoencoder(c_hid=c_hid, latent_dim=latent_dim)

    # Initialise to obtain the expected pytree structure
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, 32, 32, 3))
    variables = model.init(key, dummy)

    errors = []

    # Attempt 1 — raw msgpack bytes (most common for phlippe checkpoints)
    try:
        state = _try_load_bytes(ckpt_path, variables)
        params = state if "params" not in state else state
        print(f"[autoencoder] Loaded {ckpt_path.name} via msgpack bytes.")
        return model, params
    except Exception as e:
        errors.append(f"msgpack: {e}")

    # Attempt 2 — orbax single-file checkpoint
    try:
        state = _try_load_orbax(ckpt_path)
        params = state if isinstance(state, dict) else {"params": state}
        print(f"[autoencoder] Loaded {ckpt_path.name} via orbax.")
        return model, params
    except Exception as e:
        errors.append(f"orbax: {e}")

    # Attempt 3 — legacy flax.training.checkpoints directory
    try:
        state = _try_load_legacy(ckpt_path, variables)
        params = state
        print(f"[autoencoder] Loaded {ckpt_path.name} via legacy checkpoints.")
        return model, params
    except Exception as e:
        errors.append(f"legacy: {e}")

    raise RuntimeError(
        f"Failed to load checkpoint {ckpt_path}.\n"
        "Errors tried:\n" + "\n".join(f"  {e}" for e in errors) + "\n\n"
        "Run inspect_checkpoint(path) to view the raw checkpoint structure."
    )


def inspect_checkpoint(ckpt_path: str) -> None:
    """
    Print the top-level keys and array shapes of a checkpoint.
    Useful for debugging architecture mismatches.
    """
    import flax
    path = Path(ckpt_path)
    with open(path, "rb") as f:
        raw = flax.serialization.from_bytes(None, f.read())

    def _print_tree(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                _print_tree(v, prefix + f".{k}")
        elif hasattr(d, "shape"):
            print(f"{prefix}: {d.shape} {d.dtype}")
        else:
            print(f"{prefix}: {type(d)}")

    print(f"\n=== Checkpoint structure: {path.name} ===")
    _print_tree(raw)


# ── Encoding / decoding helpers ───────────────────────────────────────────────

def encode_dataset(
    model: Autoencoder,
    params: dict,
    images_np: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Encode a numpy array of CIFAR-10 images to latents.

    Parameters
    ----------
    images_np : float32 array of shape (N, 32, 32, 3), values in [-1, 1]

    Returns
    -------
    latents : float32 array of shape (N, latent_dim)
    """
    N = images_np.shape[0]
    latents = []
    for start in tqdm(range(0, N, batch_size), desc="Encoding", leave=False):
        batch = jnp.array(images_np[start: start + batch_size])
        z = model.apply(params, batch, method=model.encode)
        latents.append(np.array(z))
    return np.concatenate(latents, axis=0)


def decode_latents(
    model: Autoencoder,
    params: dict,
    latents_np: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Decode latent vectors back to images.

    Parameters
    ----------
    latents_np : float32 array of shape (N, latent_dim)

    Returns
    -------
    images : uint8 array of shape (N, 32, 32, 3), values in [0, 255]
    """
    N = latents_np.shape[0]
    images = []
    for start in tqdm(range(0, N, batch_size), desc="Decoding", leave=False):
        batch = jnp.array(latents_np[start: start + batch_size])
        x = model.apply(params, batch, method=model.decode)
        # tanh output in [-1, 1] → uint8 [0, 255]
        x = np.array(x)
        x = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        images.append(x)
    return np.concatenate(images, axis=0)
