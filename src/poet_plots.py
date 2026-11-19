"""Plots for the poem model (training curve + before/after evaluation)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import REPORTS_DIR

BEFORE, AFTER = "#9aa5b1", "#3b5bdb"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 11})


def plot_loss():
    df = pd.read_csv(REPORTS_DIR / "poet_loss.csv")
    tr = df.dropna(subset=["train_loss"])
    va = df.dropna(subset=["val_loss"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(tr["step"], tr["train_loss"], color=AFTER, alpha=0.25, lw=1)
    ax.plot(tr["step"], tr["train_loss"].rolling(5, min_periods=1).mean(), color=AFTER, lw=2, label="train loss (smoothed)")
    ax.plot(va["step"], va["val_loss"], "o-", color="#d0703c", lw=2, ms=4, label="validation loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"Poem model training: val loss {va['val_loss'].iloc[0]:.2f} → {va['val_loss'].iloc[-1]:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "poet_loss.png", dpi=130)
    plt.close(fig)


def plot_eval():
    df = pd.read_csv(REPORTS_DIR / "poet_eval.csv")
    metrics = {"lines": "exactly 8 lines", "rhyme": "rhyming pairs", "keyword": "fits the image\n(caption words)",
               "mood": "mood words"}
    g = df.groupby("model")[list(metrics)].mean()
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for off, (name, color) in zip((-0.2, 0.2), (("base", BEFORE), ("fine-tuned", AFTER))):
        vals = g.loc[name].values
        bars = ax.bar(x + off, vals, 0.4, color=color, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax.set_xticks(x, metrics.values())
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score (0–1, higher = better)")
    ax.set_title("Before vs after training (unseen test images)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "poet_before_after.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(2, 17) - 0.5
    for name, color in (("base", BEFORE), ("fine-tuned", AFTER)):
        ax.hist(df[df.model == name]["n_lines"], bins=bins, alpha=0.7, color=color, label=name)
    ax.axvline(8, color="#d0703c", ls="--", lw=1.5, label="target: 8")
    ax.set_xlabel("lines in generated poem")
    ax.set_ylabel("poems")
    ax.set_title("Poem length: does the model follow '8 lines'?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "poet_line_counts.png", dpi=130)
    plt.close(fig)

    # CLIP: cosine similarity between image and poem text (raw, ~0.15-0.35)
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df[df.model == m]["clip"] for m in ("base", "fine-tuned")]
    bp = ax.boxplot(data, labels=["base", "fine-tuned"], patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], (BEFORE, AFTER)):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_ylabel("CLIP cosine(image, poem)")
    ax.set_title("Does the poem match the picture?")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "poet_clip.png", dpi=130)
    plt.close(fig)
