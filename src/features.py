"""Frozen CLIP image embeddings (no training here, only feature extraction).

Run:  python -m src.features      -> builds models/features.npz from data/
"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils import DATA_DIR, FEATURES_FILE, IMG_EXT, MODELS_DIR, MOODS, timer

CLIP_NAME = "openai/clip-vit-base-patch32"
_clip = None  # loaded once, reused


def load_clip():
    global _clip
    if _clip is None:
        with timer("load CLIP"):
            model = CLIPModel.from_pretrained(CLIP_NAME).eval()
            proc = CLIPProcessor.from_pretrained(CLIP_NAME)
            _clip = (model, proc)
    return _clip


@torch.no_grad()
def embed_images(images, batch_size=16):
    """List of PIL images -> (n, 512) numpy array, L2-normalised."""
    model, proc = load_clip()
    out = []
    for i in range(0, len(images), batch_size):
        batch = [im.convert("RGB") for im in images[i:i + batch_size]]
        inputs = proc(images=batch, return_tensors="pt")
        f = model.get_image_features(**inputs)
        f = f / f.norm(dim=-1, keepdim=True)   # unit length -> fair distances
        out.append(f.numpy())
    return np.vstack(out)


def list_dataset():
    """Return (paths, labels) from data/<mood>/*.jpg."""
    paths, labels = [], []
    for mood in MOODS:
        for p in sorted((DATA_DIR / mood).glob("*")):
            if p.suffix.lower() in IMG_EXT:
                paths.append(p)
                labels.append(mood)
    return paths, labels


def main():
    paths, labels = list_dataset()
    if not paths:
        raise SystemExit("No images in data/<mood>/. Run: python -m src.build_dataset")
    print("Images per mood:", {m: labels.count(m) for m in MOODS})

    images, keep = [], []
    for p, y in zip(paths, labels):
        try:
            images.append(Image.open(p).convert("RGB"))
            keep.append(y)
        except Exception:
            print("  skip broken:", p.name)

    with timer(f"embed {len(images)} images"):
        X = embed_images(images)
    MODELS_DIR.mkdir(exist_ok=True)
    np.savez(FEATURES_FILE, X=X, y=np.array(keep))
    print(f"Saved {X.shape} features -> {FEATURES_FILE}")


if __name__ == "__main__":
    main()
