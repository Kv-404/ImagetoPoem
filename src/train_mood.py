"""Train + compare 5 scikit-learn models that predict mood from CLIP features.

Run:  python -m src.train_mood
Needs models/features.npz (python -m src.features).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import (clustering_report, plot_confusion, plot_model_comparison,
                          plot_pca_variance, scores)
from src.utils import FEATURES_FILE, MODEL_FILE, MODELS_DIR, PCA_FILE, REPORTS_DIR, timer

SEED = 42
PCA_VARIANCE = 0.90  # keep enough components for 90% of the variance

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=1000,
                         random_state=SEED),
}


def make_pipeline(clf):
    """Preprocessing (scale) -> PCA -> classifier.
    Inside a Pipeline, scaler/PCA are re-fit on each CV fold = no data leakage."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_VARIANCE, random_state=SEED)),
        ("clf", clf),
    ])


def main():
    if not FEATURES_FILE.exists():
        raise SystemExit(f"Missing {FEATURES_FILE}. Run: python -m src.features")
    data = np.load(FEATURES_FILE)
    X, y = data["X"], data["y"]
    print(f"Data: {X.shape[0]} images x {X.shape[1]} features")
    moods, counts = np.unique(y, return_counts=True)
    if len(moods) < 2 or counts.min() < 10:
        raise SystemExit(f"Need >=10 images for each mood, have {dict(zip(moods, counts))}")

    # 80/20 split, stratified so each mood keeps the same share.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, clf in MODELS.items():
        pipe = make_pipeline(clf)
        with timer(f"{name}: 5-fold CV + fit"):
            cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")
            pipe.fit(X_tr, y_tr)
        s = scores(y_te, pipe.predict(X_te))
        rows.append({"model": name, "cv_acc_mean": cv_acc.mean(), "cv_acc_std": cv_acc.std(),
                     **{f"test_{k}": v for k, v in s.items()}})
        fitted[name] = pipe
        plot_confusion(y_te, pipe.predict(X_te), name)

    df = pd.DataFrame(rows).sort_values("cv_acc_mean", ascending=False)
    pd.set_option("display.width", 120)
    print("\n", df.round(3).to_string(index=False))
    df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    plot_model_comparison(df)

    # Pick best by CV accuracy (test set stays an honest final check).
    best_name = df.iloc[0]["model"]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    print(classification_report(y_te, best.predict(X_te), zero_division=0))

    pca = best.named_steps["pca"]
    print(f"PCA: 512 -> {pca.n_components_} components "
          f"({pca.explained_variance_ratio_.sum():.1%} variance kept)")
    plot_pca_variance(pca)

    # Refit best pipeline on ALL data for the demo, then save its parts.
    final = make_pipeline(MODELS[best_name]).fit(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final[:-1], PCA_FILE)  # scaler + PCA
    joblib.dump({"model": final.named_steps["clf"], "name": best_name}, MODEL_FILE)
    print(f"Saved {PCA_FILE.name} and {MODEL_FILE.name} in models/")

    clustering_report()
    print("Plots in reports/")


if __name__ == "__main__":
    main()
