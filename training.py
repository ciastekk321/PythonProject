from __future__ import annotations
import os, re, requests
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ML libs
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

# Constants
RANDOM_STATE = 42
RAW_URL = (
    "https://raw.githubusercontent.com/Zenon-Nowakowski/Phishing-email-detection-AI/"
    "main/Phishing_Email.csv"
)
CSV_TARGET = Path("emails.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def fetch_dataset(url: str = RAW_URL, out_csv: Path = CSV_TARGET) -> Path:
    if out_csv.exists():
        print(f"✅ Plik {out_csv} już istnieje – pomijam pobieranie.")
        return out_csv
    print("⬇️  Pobieram dataset z GitHub…")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_csv.write_bytes(r.content)
    print(f"✅ Zapisano {out_csv} ({out_csv.stat().st_size // 1024} KB)")
    return out_csv

def basic_clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def prepare_data(df: pd.DataFrame):
    df = df.dropna(subset=["Email Text", "Email Type"])
    df = df.rename(columns={"Email Text": "text", "Email Type": "label"})
    df["label"] = df["label"].map({"Safe Email": 0, "Phishing Email": 1})
    df["clean_text"] = df["text"].astype(str).apply(basic_clean)
    X = df["clean_text"]
    y = df["label"]
    return train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

def build_advanced_lr_pipeline():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000))
    ])
    param_grid = {
        "tfidf__max_df": [0.9, 0.95, 1.0],
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df": [1, 5],
        "clf__penalty": ["l2"],  # use l2 for stability
        "clf__C": [0.01, 0.1, 1, 10]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    return grid

def main():
    # Load data
    csv_path = fetch_dataset()
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Build and tune LR
    print("🔧 Buduję i wyszukuję hiperparametry dla LR…")
    grid = build_advanced_lr_pipeline()
    grid.fit(X_train, y_train)
    print(f"🔎 Najlepsze parametry: {grid.best_params_}")

    # Calibrate probabilities
    print("⚖️ Kalibruję model…")
    calibrated = CalibratedClassifierCV(grid.best_estimator_, cv='prefit', method='isotonic')
    calibrated.fit(X_train, y_train)

    # Evaluate
    print("📝 Ewaluacja na zbiorze testowym")
    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")

    # Plot and save
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix – LR")
    plt.savefig(OUT_DIR / "cm_lr.png", dpi=150)
    plt.clf()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve – LR")
    plt.savefig(OUT_DIR / "roc_lr.png", dpi=150)
    plt.clf()

    # Cross-validation stability
    print("📊 Cross-validation scores (ROC AUC):")
    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"Średni ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save final model
    joblib.dump(calibrated, OUT_DIR / "model_LR_advanced.joblib")
    print(f"✅ Zapisano model w {OUT_DIR / 'model_LR_advanced.joblib'}")

if __name__ == "__main__":
    main()