"""Load the trained mood model and predict mood for one image."""
import joblib

from src.features import embed_images
from src.utils import MODEL_FILE, PCA_FILE


def load_mood_model():
    """Return (pca_pipeline, classifier, name). Clear error if not trained yet."""
    missing = [p.name for p in (PCA_FILE, MODEL_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Model file(s) missing in models/: {', '.join(missing)}.\n"
            "Train first:\n"
            "  python -m src.build_dataset   # get images\n"
            "  python -m src.features        # CLIP embeddings\n"
            "  python -m src.train_mood      # train + save model")
    pca = joblib.load(PCA_FILE)
    saved = joblib.load(MODEL_FILE)
    return pca, saved["model"], saved["name"]


def predict_mood(image, pca, clf):
    """PIL image -> (best mood, confidence, {mood: prob})."""
    feats = embed_images([image])          # (1, 512) CLIP vector
    z = pca.transform(feats)               # scale + PCA
    probs = clf.predict_proba(z)[0]
    all_probs = dict(zip(map(str, clf.classes_), probs.round(3).tolist()))
    i = probs.argmax()
    return str(clf.classes_[i]), float(probs[i]), all_probs
