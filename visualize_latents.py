import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


LATENT_DIMS = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES = [50_000, 100_000, 150_000, 200_000]

LATENT_DIR = Path("latents")
RESULTS_DIR = Path("results/trained_AE")
OUT_DIR = Path("results/teacher_latent_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)



def load_real_latents(dim: int) -> np.ndarray:
    return np.load(LATENT_DIR / f"latents_{dim}.npy").astype(np.float32)


def load_teacher_latents(dim: int) -> np.ndarray:
    path = RESULTS_DIR / f"z_orig_{dim}_teacher.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run Step 4 --generate --teacher for dim={dim} first."
        )
    return np.load(path).astype(np.float32)


def load_student_latents(dim: int, size: int) -> np.ndarray | None:
    path = RESULTS_DIR / f"z_orig_{dim}_{size}.npy"
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def sample_same_size(real: np.ndarray, gen: np.ndarray, seed: int = 0):
    n = min(len(real), len(gen))
    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real), size=n, replace=False)
    gen_idx = rng.choice(len(gen), size=n, replace=False)
    return real[real_idx], gen[gen_idx]


def diagonal_gaussian_stats(real: np.ndarray, gen: np.ndarray, eps: float = 1e-8):
    real_mean = real.mean(axis=0)
    gen_mean = gen.mean(axis=0)

    real_std = real.std(axis=0) + eps
    gen_std = gen.std(axis=0) + eps

    mean_l2 = np.linalg.norm(real_mean - gen_mean)
    mean_l2_per_dim = mean_l2 / np.sqrt(real.shape[1])

    log_std_diff = np.abs(np.log(gen_std) - np.log(real_std))

    std_ratio = gen_std / real_std
    std_ratio_median = np.median(std_ratio)
    std_ratio_p10 = np.percentile(std_ratio, 10)
    std_ratio_p90 = np.percentile(std_ratio, 90)

    return {
        "mean_l2": float(mean_l2),
        "mean_l2_per_sqrt_dim": float(mean_l2_per_dim),
        "mean_abs_log_std_diff": float(log_std_diff.mean()),
        "median_std_ratio_gen_over_real": float(std_ratio_median),
        "p10_std_ratio": float(std_ratio_p10),
        "p90_std_ratio": float(std_ratio_p90),
    }


def covariance_spectrum_stats(real: np.ndarray, gen: np.ndarray, n_components: int = 50):
    scaler_mean = real.mean(axis=0, keepdims=True)
    scaler_std = real.std(axis=0, keepdims=True) + 1e-8

    real_z = (real - scaler_mean) / scaler_std
    gen_z = (gen - scaler_mean) / scaler_std

    pca_real = PCA(n_components=n_components, random_state=0).fit(real_z)
    pca_gen = PCA(n_components=n_components, random_state=0).fit(gen_z)

    real_ev = pca_real.explained_variance_ratio_
    gen_ev = pca_gen.explained_variance_ratio_

    return {
        "real_top1_var": float(real_ev[0]),
        "real_top10_var": float(real_ev[:10].sum()),
        "real_top50_var": float(real_ev[:50].sum()),
        "gen_top1_var": float(gen_ev[0]),
        "gen_top10_var": float(gen_ev[:10].sum()),
        "gen_top50_var": float(gen_ev[:50].sum()),
        "abs_top50_spectrum_diff": float(np.abs(real_ev - gen_ev).sum()),
        "real_spectrum": real_ev.tolist(),
        "gen_spectrum": gen_ev.tolist(),
    }


def mmd_rbf(real: np.ndarray, gen: np.ndarray, max_n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = min(len(real), len(gen), max_n)

    real = real[rng.choice(len(real), size=n, replace=False)]
    gen = gen[rng.choice(len(gen), size=n, replace=False)]

    # standardize using real statistics
    mean = real.mean(axis=0, keepdims=True)
    std = real.std(axis=0, keepdims=True) + 1e-8
    x = (real - mean) / std
    y = (gen - mean) / std

    # median heuristic on subset
    xy = np.concatenate([x[:1000], y[:1000]], axis=0)
    dists = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    median_sq = np.median(dists[dists > 0])
    gamma = 1.0 / (2.0 * median_sq + 1e-8)

    def kernel(a, b):
        d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d)

    kxx = kernel(x, x).mean()
    kyy = kernel(y, y).mean()
    kxy = kernel(x, y).mean()

    return float(kxx + kyy - 2.0 * kxy)

from sklearn.manifold import TSNE


def _size_label(size: int) -> str:
    return f"{size // 1_000_000}M" if size >= 1_000_000 else f"{size // 1_000}k"


def plot_2d_comparison(
    X_a: np.ndarray,
    X_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    out_name: str,
    method: str = "tsne",
    n: int = 2000,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = min(n, len(X_a), len(X_b))

    a_s = X_a[rng.choice(len(X_a), size=n, replace=False)]
    b_s = X_b[rng.choice(len(X_b), size=n, replace=False)]

    X = np.concatenate([a_s, b_s], axis=0)
    y = np.array([0] * n + [1] * n)
    X = (X - a_s.mean(axis=0, keepdims=True)) / (a_s.std(axis=0, keepdims=True) + 1e-8)

    if method == "tsne":
        xy = TSNE(
            n_components=2,
            perplexity=40,
            max_iter=1000,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        ).fit_transform(X)
    elif method == "umap":
        import umap
        xy = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            random_state=seed,
        ).fit_transform(X)
    else:
        raise ValueError(method)

    plt.figure(figsize=(7, 6))
    plt.scatter(xy[y == 0, 0], xy[y == 0, 1], s=5, alpha=0.45, label=label_a)
    plt.scatter(xy[y == 1, 0], xy[y == 1, 1], s=5, alpha=0.45, label=label_b)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=150)
    plt.close()


def plot_2d_real_vs_teacher(dim, real, gen, method="tsne", n=2000, seed=0):
    plot_2d_comparison(
        real, gen,
        label_a="real CIFAR latents",
        label_b="teacher latents",
        title=f"{method.upper()} real vs teacher — dim={dim}",
        out_name=f"{method}_real_vs_teacher_dim_{dim}.png",
        method=method,
        n=n,
        seed=seed,
    )


def plot_2d_teacher_vs_student(dim, teacher, student, size, method="tsne", n=2000, seed=0):
    lbl = _size_label(size)
    plot_2d_comparison(
        teacher, student,
        label_a="teacher latents",
        label_b=f"student latents ({lbl})",
        title=f"{method.upper()} teacher vs student ({lbl}) — dim={dim}",
        out_name=f"{method}_teacher_vs_student_{size}_dim_{dim}.png",
        method=method,
        n=n,
        seed=seed,
    )


def plot_2d_real_vs_student(dim, real, student, size, method="tsne", n=2000, seed=0):
    lbl = _size_label(size)
    plot_2d_comparison(
        real, student,
        label_a="real CIFAR latents",
        label_b=f"student latents ({lbl})",
        title=f"{method.upper()} real vs student ({lbl}) — dim={dim}",
        out_name=f"{method}_real_vs_student_{size}_dim_{dim}.png",
        method=method,
        n=n,
        seed=seed,
    )


def plot_norm_hist(dim: int, real: np.ndarray, gen: np.ndarray):
    real_norm = np.linalg.norm(real, axis=1)
    gen_norm = np.linalg.norm(gen, axis=1)

    plt.figure(figsize=(8, 5))
    plt.hist(real_norm, bins=80, alpha=0.5, density=True, label="real CIFAR latents")
    plt.hist(gen_norm, bins=80, alpha=0.5, density=True, label="teacher generated latents")
    plt.title(f"Latent norm distribution — dim={dim}")
    plt.xlabel("||z||")
    plt.ylabel("density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"norm_hist_dim_{dim}.png", dpi=150)
    plt.close()


def plot_pca_spectrum(dim: int, spec):
    real = np.array(spec["real_spectrum"])
    gen = np.array(spec["gen_spectrum"])

    plt.figure(figsize=(8, 5))
    plt.plot(real, marker="o", label="real CIFAR latents")
    plt.plot(gen, marker="o", label="teacher generated latents")
    plt.title(f"PCA spectrum comparison — dim={dim}")
    plt.xlabel("principal component")
    plt.ylabel("explained variance ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"pca_spectrum_dim_{dim}.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["pca", "tsne", "umap", "all"], default="all")
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--dims", type=int, nargs="+", default=LATENT_DIMS)
    parser.add_argument("--sizes", type=int, nargs="+", default=DATASET_SIZES)
    args = parser.parse_args()

    summary = {}

    for dim in args.dims:
        print(f"\n=== dim={dim} ===")

        real = load_real_latents(dim)
        gen = load_teacher_latents(dim)
        real, gen = sample_same_size(real, gen)

        diag = diagonal_gaussian_stats(real, gen)
        spec = covariance_spectrum_stats(real, gen)
        mmd = mmd_rbf(real, gen)

        summary[str(dim)] = {
            "diagonal_stats": diag,
            "pca_stats": {k: v for k, v in spec.items() if not k.endswith("spectrum")},
            "mmd_rbf": mmd,
        }

        print(json.dumps(summary[str(dim)], indent=2))

        if args.method in ["pca", "all"]:
            plot_norm_hist(dim, real, gen)
            plot_pca_spectrum(dim, spec)

        do_tsne = args.method in ["tsne", "all"]
        do_umap = args.method in ["umap", "all"]

        # ── real vs teacher ───────────────────────────────────────────────────
        if do_tsne:
            print(f"  TSNE real vs teacher dim={dim} ...")
            plot_2d_real_vs_teacher(dim, real, gen, method="tsne", n=args.n_samples)

        if do_umap:
            try:
                print(f"  UMAP real vs teacher dim={dim} ...")
                plot_2d_real_vs_teacher(dim, real, gen, method="umap", n=args.n_samples)
            except ModuleNotFoundError:
                print("  [warning] umap-learn not installed — skipping UMAP.")

        # ── teacher vs student  /  real vs student (per size) ─────────────────
        for size in args.sizes:
            student = load_student_latents(dim, size)
            if student is None:
                print(f"  [skip] student size={size} not found for dim={dim}")
                continue

            lbl = _size_label(size)

            if do_tsne:
                print(f"  TSNE teacher vs student ({lbl}) dim={dim} ...")
                plot_2d_teacher_vs_student(dim, gen, student, size, method="tsne", n=args.n_samples)
                print(f"  TSNE real vs student ({lbl}) dim={dim} ...")
                plot_2d_real_vs_student(dim, real, student, size, method="tsne", n=args.n_samples)

            if do_umap:
                try:
                    print(f"  UMAP teacher vs student ({lbl}) dim={dim} ...")
                    plot_2d_teacher_vs_student(dim, gen, student, size, method="umap", n=args.n_samples)
                    print(f"  UMAP real vs student ({lbl}) dim={dim} ...")
                    plot_2d_real_vs_student(dim, real, student, size, method="umap", n=args.n_samples)
                except ModuleNotFoundError:
                    print("  [warning] umap-learn not installed — skipping UMAP.")

    out_json = OUT_DIR / "teacher_latent_distribution_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary → {out_json}")
    print(f"Saved plots   → {OUT_DIR}/")


if __name__ == "__main__":
    main()