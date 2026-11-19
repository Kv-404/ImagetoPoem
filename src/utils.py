"""Shared paths, mood list and a tiny timer for the demo."""
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # hide fork warning
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FEATURES_FILE = MODELS_DIR / "features.npz"   # cached CLIP embeddings
PCA_FILE = MODELS_DIR / "pca.joblib"           # StandardScaler + PCA
MODEL_FILE = MODELS_DIR / "mood_model.joblib"  # best classifier

MOODS = ["happy", "sad", "calm", "energetic"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@contextmanager
def timer(step):
    """Print how long a step took:  with timer("caption"): ..."""
    start = time.perf_counter()
    yield
    print(f"[time] {step}: {time.perf_counter() - start:.2f}s")


def best_device():
    """GPU if we have one: NVIDIA (cuda, e.g. Colab) > Apple (mps) > cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
