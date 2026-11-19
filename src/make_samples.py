"""Run the full pipeline on 1 image per mood -> site/samples.json for the website.

Run:  python -m src.make_samples   (after training)
"""
import json
import random
import time

from PIL import Image

from src.caption import caption_image
from src.features import list_dataset
from src.mood import load_mood_model, predict_mood
from src.poem import write_poem
from src.utils import MOODS, ROOT

SITE = ROOT / "site"
OUT_DIR = SITE / "samples"


def main():
    pca, clf, _ = load_mood_model()
    paths, labels = list_dataset()
    random.seed(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for mood in MOODS:
        path = random.choice([p for p, y in zip(paths, labels) if y == mood])
        img = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        caption = caption_image(img)
        t1 = time.perf_counter()
        pred, conf, _ = predict_mood(img, pca, clf)
        t2 = time.perf_counter()
        poem = write_poem(caption, pred)
        t3 = time.perf_counter()
        timing = {"caption": round(t1 - t0, 2), "mood": round(t2 - t1, 2), "poem": round(t3 - t2, 2)}
        print(f"[time] sample {mood}: {t3 - t0:.2f}s {timing}")
        img.thumbnail((480, 480))
        name = f"{mood}.jpg"
        img.save(OUT_DIR / name, quality=85)
        samples.append({"image": f"samples/{name}", "caption": caption,
                        "mood": pred, "confidence": round(conf, 3), "poem": poem, "timing": timing})
    (SITE / "samples.json").write_text(json.dumps(samples, indent=2))
    print(f"Saved {len(samples)} samples -> site/samples.json")


if __name__ == "__main__":
    main()
