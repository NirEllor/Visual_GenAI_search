import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


LATENT_DIMS = [64, 128, 256, 384, 512, 1024]
DATASET_SIZES = [50_000, 100_000, 150_000, 200_000]

LATENT_DIR = Path("latents")
RESULTS_DIR = Path("results/trained_AE")
OUT_DIR = Path("results/student_latent_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_real_latents(dim: int) -> np.ndarray:
    return np.load(LATENT_DIR / f"latents_{dim}.npy").astype(np.float32)


def load_generated_latents(dim: int, tag: str) -> np.ndarray:
    path = RESULTS_DIR / f"z_orig_{dim}_{tag}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run Step 4 --generate first.")
    return np.load(path).astype(np.float32)


def sample_same_size(a: np.ndarray, b: np.ndarray, seed: int = 0):
    n = min(len(a), len(b))
    rng = np.random.default_rng(seed)
    a_idx = rng.choice(len(a), size=n, replace=False)
    b_idx = rng.choice(len(b), size=n, replace=False)
    return a[a_idx], b[b_idx]


def diagonal_gaussian_stats(real: np.ndarray, gen: np.ndarray, eps: float = 1e-8):
    real_mean = real.mean(axis=0)
    gen_mean = gen.mean(axis=0)

    real_std = real.std(axis=0) + eps
    gen_std = gen.std(axis=0) + eps

    std_ratio = gen_std / real_std
    log_std_diff = np.abs(np.log(gen_std) - np.log(real_std))

    return {
        "mean_l2": float(np.linalg.norm(real_mean - gen_mean)),
        "mean_l2_per_sqrt_dim": float(np.linalg.norm(real_mean - gen_mean) / np.sqrt(real.shape[1])),
        "mean_abs_log_std_diff": float(log_std_diff.mean()),
        "median_std_ratio_gen_over_real": float(np.median(std_ratio)),
        "p10_std_ratio": float(np.percentile(std_ratio, 10)),
        "p90_std_ratio": float(np.percentile(std_ratio, 90)),
    }


def covariance_spectrum_stats(real: np.ndarray, gen: np.ndarray, n_components: int = 50):
    mean = real.mean(axis=0, keepdims=True)
    std = real.std(axis=0, keepdims=True) + 1e-8

    real_z = (real - mean) / std
    gen_z = (gen - mean) / std

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

    mean = real.mean(axis=0, keepdims=True)
    std = real.std(axis=0, keepdims=True) + 1e-8

    x = (real - mean) / std
    y = (gen - mean) / std

    xy = np.concatenate([x[:1000], y[:1000]], axis=0)
    dists = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    median_sq = np.median(dists[dists > 0])
    gamma = 1.0 / (2.0 * median_sq + 1e-8)

    def kernel(a, b):
        d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d)

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def norm_stats(arr: np.ndarray):
    norms = np.linalg.norm(arr, axis=1)
    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_p1": float(np.percentile(norms, 1)),
        "norm_p10": float(np.percentile(norms, 10)),
        "norm_p50": float(np.percentile(norms, 50)),
        "norm_p90": float(np.percentile(norms, 90)),
        "norm_p99": float(np.percentile(norms, 99)),
        "frac_norm_lt_1": float((norms < 1.0).mean()),
        "frac_norm_lt_5": float((norms < 5.0).mean()),
        "frac_norm_lt_10": float((norms < 10.0).mean()),
    }


def plot_norm_hist(dim: int, tag: str, real: np.ndarray, gen: np.ndarray):
    real_norm = np.linalg.norm(real, axis=1)
    gen_norm = np.linalg.norm(gen, axis=1)

    plt.figure(figsize=(8, 5))
    plt.hist(real_norm, bins=80, alpha=0.5, density=True, label="real CIFAR latents")
    plt.hist(gen_norm, bins=80, alpha=0.5, density=True, label=f"{tag} latents")
    plt.title(f"Latent norm distribution — dim={dim}, {tag}")
    plt.xlabel("||z||")
    plt.ylabel("density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"norm_hist_dim_{dim}_{tag}.png", dpi=150)
    plt.close()


def plot_pca_spectrum(dim: int, tag: str, spec):
    real = np.array(spec["real_spectrum"])
    gen = np.array(spec["gen_spectrum"])

    plt.figure(figsize=(8, 5))
    plt.plot(real, marker="o", label="real CIFAR latents")
    plt.plot(gen, marker="o", label=f"{tag} latents")
    plt.title(f"PCA spectrum comparison — dim={dim}, {tag}")
    plt.xlabel("principal component")
    plt.ylabel("explained variance ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"pca_spectrum_dim_{dim}_{tag}.png", dpi=150)
    plt.close()


def plot_2d(dim: int, tag: str, real: np.ndarray, gen: np.ndarray, method: str, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = min(n, len(real), len(gen))

    real_s = real[rng.choice(len(real), size=n, replace=False)]
    gen_s = gen[rng.choice(len(gen), size=n, replace=False)]

    X = np.concatenate([real_s, gen_s], axis=0)
    y = np.array([0] * n + [1] * n)

    X = (X - real_s.mean(axis=0, keepdims=True)) / (real_s.std(axis=0, keepdims=True) + 1e-8)

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
    plt.scatter(xy[y == 0, 0], xy[y == 0, 1], s=5, alpha=0.45, label="real CIFAR latents")
    plt.scatter(xy[y == 1, 0], xy[y == 1, 1], s=5, alpha=0.45, label=f"{tag} latents")
    plt.title(f"{method.upper()} real vs {tag} — dim={dim}")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{method}_real_vs_{tag}_dim_{dim}.png", dpi=150)
    plt.close()


def analyze_pair(dim: int, tag: str, real: np.ndarray, gen: np.ndarray, method: str, n_samples: int):
    real_s, gen_s = sample_same_size(real, gen)

    diag = diagonal_gaussian_stats(real_s, gen_s)
    spec = covariance_spectrum_stats(real_s, gen_s)
    mmd = mmd_rbf(real_s, gen_s)

    result = {
        "diagonal_stats": diag,
        "pca_stats": {k: v for k, v in spec.items() if not k.endswith("spectrum")},
        "mmd_rbf": mmd,
        "real_norm_stats": norm_stats(real_s),
        "gen_norm_stats": norm_stats(gen_s),
    }

    plot_norm_hist(dim, tag, real_s, gen_s)
    plot_pca_spectrum(dim, tag, spec)

    if method in ["tsne", "all"]:
        plot_2d(dim, tag, real_s, gen_s, method="tsne", n=n_samples)

    if method in ["umap", "all"]:
        try:
            plot_2d(dim, tag, real_s, gen_s, method="umap", n=n_samples)
        except ModuleNotFoundError:
            print("  [warning] umap-learn not installed — skipping UMAP.")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=LATENT_DIMS)
    parser.add_argument("--sizes", type=int, nargs="+", default=DATASET_SIZES)
    parser.add_argument("--include-teacher", action="store_true")
    parser.add_argument("--method", choices=["pca", "tsne", "umap", "all"], default="pca")
    parser.add_argument("--n-samples", type=int, default=3000)
    args = parser.parse_args()

    summary = {}

    for dim in args.dims:
        print(f"\n=== dim={dim} ===")
        real = load_real_latents(dim)

        dim_summary = {}

        if args.include_teacher:
            print("  analyzing teacher")
            teacher = load_generated_latents(dim, "teacher")
            dim_summary["teacher"] = analyze_pair(
                dim, "teacher", real, teacher, args.method, args.n_samples
            )

        for size in args.sizes:
            tag = str(size)
            print(f"  analyzing student {tag}")

            try:
                student = load_generated_latents(dim, tag)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
                continue

            dim_summary[tag] = analyze_pair(
                dim, f"student_{tag}", real, student, args.method, args.n_samples
            )

            print(json.dumps(dim_summary[tag], indent=2))

        summary[str(dim)] = dim_summary

    out_json = OUT_DIR / "student_latent_distribution_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary → {out_json}")
    print(f"Saved plots   → {OUT_DIR}/")


if __name__ == "__main__":
    main()