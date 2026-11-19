"""Download ~50 Creative Commons images per mood from Openverse (no API key).

Run:  python -m src.build_dataset --per-mood 60
Then LOOK at data/<mood>/ and delete images that do not fit the mood.
Clean labels matter more than more images.
"""
import argparse
import io
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

from src.utils import DATA_DIR, MOODS

API = "https://api.openverse.org/v1/images/"
HEADERS = {"User-Agent": "image-to-poem-student-project"}

# Several search words per mood -> more variety in the data.
QUERIES = {
    "happy": ["smiling people", "birthday party", "sunflower field", "children laughing", "colorful balloons"],
    "sad": ["rainy window", "lonely person", "gray rainy street", "wilted flower", "abandoned house"],
    "calm": ["calm lake", "misty forest", "zen garden", "quiet beach sunset", "still water reflection"],
    "energetic": ["concert crowd", "running athlete", "fireworks", "skateboarding trick", "dance party"],
}


def search(query, page):
    params = {"q": query, "page_size": 20, "page": page, "mature": "false"}
    r = requests.get(API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()["results"]


def save_image(url, path):
    """Download, check it is a real image, shrink to 512px, save as JPG."""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img.thumbnail((512, 512))
    img.save(path, "JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-mood", type=int, default=60)
    args = ap.parse_args()

    for mood in MOODS:
        out = DATA_DIR / mood
        out.mkdir(parents=True, exist_ok=True)
        need = args.per_mood - len(list(out.glob("*.jpg")))
        print(f"\n== {mood}: need {max(need, 0)} more")
        if need <= 0:
            continue

        # 1) collect candidate links (a few per query = more variety)
        jobs, seen = [], set()
        for page in (1, 2, 3, 4):
            for q in QUERIES[mood]:
                try:
                    results = search(q, page)
                except Exception as e:
                    print("  search failed:", q, e)
                    continue
                for res in results[:5]:
                    path = out / f"{res['id']}.jpg"
                    if res["id"] not in seen and not path.exists():
                        seen.add(res["id"])
                        jobs.append((res.get("thumbnail") or res["url"], path))
            if len(jobs) >= need * 2:      # extra, because some links die
                break

        # 2) download 8 at a time (much faster than one by one)
        def fetch(job):
            try:
                save_image(*job)
                return True
            except Exception:
                return False               # dead link / not an image: skip

        got = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i in range(0, len(jobs), 8):
                if got >= need:
                    break
                got += sum(pool.map(fetch, jobs[i:i + 8]))
        print(f"  got {got} new images")
    print("\nDone. Now open data/<mood>/ and delete wrong images.")
    print("Images are CC licensed from Openverse; keep them for study use only.")


if __name__ == "__main__":
    main()
