"""Evaluation + plots (used by train_mood) and the clustering bonus.

Run:  python -m src.evaluate    -> K-Means elbow + 2D PCA scatter in reports/
"""
import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             adjusted_rand_score, f1_score, precision_score,
                             recall_score, silhouette_score)

from src.utils import FEATURES_FILE, MOODS, REPORTS_DIR, timer

REPORTS_DIR.mkdir(exist_ok=True)


def scores(y_true, y_pred):
    """Accuracy + macro precision/recall/F1 (macro = each mood counts equally)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion(y_true, y_pred, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=MOODS, ax=ax,
                                            cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    path = REPORTS_DIR / f"confusion_{name.replace(' ', '_')}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pca_variance(pca):
    """Bar = variance per component, line = running total."""
    ratio = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(ratio) + 1), ratio, color="#7aa6d8", label="per component")
    ax.plot(range(1, len(ratio) + 1), np.cumsum(ratio), "o-", color="#d0703c",
            ms=3, label="cumulative")
    ax.axhline(0.90, ls="--", color="gray", lw=1)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title(f"PCA: {len(ratio)} components keep {ratio.sum():.0%} of variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pca_explained_variance.png", dpi=120)
    plt.close(fig)


def plot_model_comparison(df):
    """Grouped bars: CV accuracy vs test metrics for each model."""
    cols = ["cv_acc_mean", "test_accuracy", "test_f1"]
    ax = df.set_index("model")[cols].plot.bar(figsize=(8, 4), rot=20,
                                              yerr={"cv_acc_mean": df["cv_acc_std"].values})
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (5-fold CV + held-out test)")
    ax.figure.tight_layout()
    ax.figure.savefig(REPORTS_DIR / "model_comparison.png", dpi=120)
    plt.close(ax.figure)


def clustering_report():
    """Bonus: unsupervised view of the CLIP embeddings."""
    data = np.load(FEATURES_FILE)
    X, y = data["X"], data["y"]

    # Elbow: inertia for k = 1..10. Look for the "bend".
    with timer("K-Means elbow (k=1..10)"):
        ks = range(1, 11)
        inertias = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ks), inertias, "o-")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("inertia (within-cluster SSE)")
    ax.set_title("K-Means elbow plot")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "kmeans_elbow.png", dpi=120)
    plt.close(fig)

    # k=4 (same as moods): do clusters match the moods?
    km = KMeans(4, n_init=10, random_state=42).fit(X)
    print(f"k=4 silhouette: {silhouette_score(X, km.labels_):.3f}  "
          f"ARI vs moods: {adjusted_rand_score(y, km.labels_):.3f}")

    # 2D PCA scatter coloured by true mood.
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    for mood in MOODS:
        m = y == mood
        ax.scatter(X2[m, 0], X2[m, 1], s=18, alpha=0.75, label=mood)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("CLIP embeddings in 2D (PCA), coloured by mood")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pca_2d_scatter.png", dpi=120)
    plt.close(fig)
    print("Saved reports/kmeans_elbow.png and reports/pca_2d_scatter.png")


if __name__ == "__main__":
    if not FEATURES_FILE.exists():
        raise SystemExit(f"Missing {FEATURES_FILE}. Run: python -m src.features")
    clustering_report()
