"""Collect every number the website shows into site/data.json (+ small thumbnails).

Re-runs the cheap mood experiments with the same seed as train_mood.py so the
site can draw interactive charts (not static PNGs). Poem-model results are read
from the files train_poet.py / eval_poet.py wrote (skipped if missing).

Run:  python -m src.site_data   then   python site/build_site.py
"""
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluate import scores
from src.features import list_dataset
from src.train_mood import MODELS, SEED, make_pipeline
from src.utils import DATA_DIR, FEATURES_FILE, MODELS_DIR, MOODS, REPORTS_DIR, ROOT

SITE = ROOT / "site"
THUMBS = SITE / "thumbs"


def r(x, n=4):
    return round(float(x), n)


def mood_section():
    d = np.load(FEATURES_FILE)
    X, y = d["X"], d["y"]
    paths, labels = list_dataset()
    assert len(paths) == len(y) and list(labels) == list(y), "features.npz out of date: run python -m src.features"

    # thumbnails for hover previews
    THUMBS.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for i, p in enumerate(paths):
        name = f"{i:03d}.jpg"
        if not (THUMBS / name).exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((160, 160))
            im.save(THUMBS / name, quality=80)
        thumbs.append(f"thumbs/{name}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = []
    for name, clf in MODELS.items():
        pipe = make_pipeline(clf)
        t = time.perf_counter()
        folds = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")
        pipe.fit(X_tr, y_tr)
        secs = time.perf_counter() - t
        pred = pipe.predict(X_te)
        p, rc, f, sup = precision_recall_fscore_support(y_te, pred, labels=MOODS, zero_division=0)
        models.append({
            "name": name, "cv_folds": [r(v) for v in folds], "cv_mean": r(folds.mean()), "cv_std": r(folds.std()),
            "test": {k: r(v) for k, v in scores(y_te, pred).items()}, "seconds": r(secs, 2),
            "confusion": confusion_matrix(y_te, pred, labels=MOODS).tolist(),
            "per_class": {m: {"precision": r(p[i]), "recall": r(rc[i]), "f1": r(f[i]), "support": int(sup[i])}
                          for i, m in enumerate(MOODS)},
        })
    models.sort(key=lambda m: -m["cv_mean"])

    # PCA curve (fit on scaled training data, as in the pipeline)
    Z = StandardScaler().fit_transform(X_tr)
    full = PCA(random_state=SEED).fit(Z)
    cum = np.cumsum(full.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum, 0.90) + 1)

    # Unsupervised: elbow + silhouette, 2D PCA and t-SNE maps
    ks = list(range(1, 11))
    km = [KMeans(k, n_init=10, random_state=SEED).fit(X) for k in ks]
    sil = {k: r(silhouette_score(X, m.labels_)) for k, m in zip(ks, km) if k >= 2}
    from sklearn.metrics import adjusted_rand_score
    pca2 = PCA(n_components=2, random_state=SEED).fit(X)
    P2 = pca2.transform(X)
    T2 = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(X)
    points = [{"pca": [r(P2[i, 0]), r(P2[i, 1])], "tsne": [r(T2[i, 0], 2), r(T2[i, 1], 2)],
               "mood": str(y[i]), "cluster": int(km[3].labels_[i]), "thumb": thumbs[i]} for i in range(len(y))]

    return {
        "counts": {m: int((y == m).sum()) for m in MOODS},
        "n_images": int(len(y)), "n_features": int(X.shape[1]),
        "split": {"train": int(len(y_tr)), "test": int(len(y_te))},
        "models": models, "best": models[0]["name"],
        "pca": {"cumulative": [r(v) for v in cum[:150]], "ratio": [r(v, 5) for v in full.explained_variance_ratio_[:150]],
                "n90": n90, "pc2_var": [r(v) for v in pca2.explained_variance_ratio_]},
        "kmeans": {"k": ks, "inertia": [r(m.inertia_, 2) for m in km], "silhouette": sil,
                   "ari_k4": r(adjusted_rand_score(y, km[3].labels_))},
        "points": points,
    }


def poet_section():
    out = {}
    poems = DATA_DIR / "poems"
    if (poems / "train.jsonl").exists():
        rows = {s: [json.loads(l) for l in open(poems / f"{s}.jsonl")] for s in ("train", "val", "test")}
        out["dataset"] = {s: len(v) for s, v in rows.items()}
        out["examples"] = [{k: e[k] for k in ("image", "caption", "mood", "poem", "score")}
                           for e in rows["train"][::max(1, len(rows["train"]) // 6)][:6]]
        sc = poems / "scores.json"
        if sc.exists():
            s = json.load(open(sc))
            out["scores"] = {"candidates": [r(v, 3) for v in s["candidate_scores"]],
                             "kept": [r(v, 3) for v in s["kept_scores"]]}
    info = MODELS_DIR / "poet_lora" / "train_info.json"
    if info.exists():
        out["train_info"] = json.load(open(info))
    loss = REPORTS_DIR / "poet_loss.csv"
    if loss.exists():
        df = pd.read_csv(loss)
        out["loss"] = {"step": df["step"].tolist(),
                       "train": [None if pd.isna(v) else r(v) for v in df["train_loss"]],
                       "val": [None if pd.isna(v) else r(v) for v in df["val_loss"]]}
    ev = REPORTS_DIR / "poet_eval.csv"
    if ev.exists():
        df = pd.read_csv(ev)
        cols = ["lines", "rhyme", "keyword", "mood", "clip", "total"]
        out["eval"] = {m: {c: r(df[df.model == m][c].mean()) for c in cols} for m in ("base", "fine-tuned")}
        out["eval"]["n_poems"] = int((df.model == "base").sum())
        out["line_counts"] = {m: df[df.model == m]["n_lines"].value_counts().sort_index().to_dict()
                              for m in ("base", "fine-tuned")}
        out["line_counts"] = {m: {int(k): int(v) for k, v in d.items()} for m, d in out["line_counts"].items()}
        out["clip"] = {m: [r(v) for v in df[df.model == m]["clip"]] for m in ("base", "fine-tuned")}
        # side-by-side: same image, best-scoring fine-tuned vs a base poem
        pairs = []
        for img, g in df.groupby("image"):
            b, f = g[g.model == "base"], g[g.model == "fine-tuned"]
            if len(b) and len(f):
                pairs.append((f["total"].max() - b["total"].mean(), img, g))
        pairs.sort(key=lambda t: -t[0])
        out["compare"] = []
        for _, img, g in pairs[:4]:
            b = g[g.model == "base"].iloc[0]
            f = g[g.model == "fine-tuned"].sort_values("total").iloc[-1]
            out["compare"].append({"image": img, "caption": b["caption"],
                                   "mood": b.get("mood_label", ""),
                                   "base": b["poem"], "tuned": f["poem"],
                                   "base_lines": int(b["n_lines"]), "tuned_lines": int(f["n_lines"])})
    return out


def main():
    t = time.perf_counter()
    data = {"moods": MOODS, "mood": mood_section(), "poet": poet_section()}
    samples = SITE / "samples.json"
    data["samples"] = json.load(open(samples)) if samples.exists() else []
    # images used by poem examples -> thumbnails too
    for key in ("examples", "compare"):
        for e in data["poet"].get(key, []):
            src = DATA_DIR / e["image"]
            name = "p_" + src.stem[:12] + ".jpg"
            if not (THUMBS / name).exists():
                im = Image.open(src).convert("RGB")
                im.thumbnail((360, 360))
                im.save(THUMBS / name, quality=82)
            e["thumb"] = f"thumbs/{name}"
    (SITE / "data.json").write_text(json.dumps(data))
    print(f"Wrote site/data.json ({(SITE / 'data.json').stat().st_size // 1024} KB) in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
