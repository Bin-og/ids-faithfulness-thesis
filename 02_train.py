"""
02_train.py
-----------
Trains RandomForest and XGBoost on the IDS2018 training split, then
evaluates both models two ways:
  (a) in-distribution:  IDS2018 held-out test split
  (b) cross-dataset:    CICIDS2017 test split (same trained model, no retraining)

This is where you should see the accuracy/MCC collapse that motivates
RQ1/RQ2 (cross-dataset generalization failure).

Run: python scripts/02_train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_split(name):
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return None, None, None
    df = pd.read_parquet(path)
    with open(DATA_DIR / "feature_columns.json") as f:
        feat_cols = json.load(f)
    X = df[feat_cols].values
    y = df["Label"].values
    return X, y, feat_cols


def evaluate(model, X, y, name, results):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    mcc = matthews_corrcoef(y, pred)
    print(f"\n--- {name} ---")
    print(f"accuracy={acc:.4f}  f1={f1:.4f}  mcc={mcc:.4f}")
    print(classification_report(y, pred, target_names=["benign", "attack"]))
    results[name] = {"accuracy": acc, "f1": f1, "mcc": mcc}


def main():
    X_train, y_train, feat_cols = load_split("ids2018_train")
    X_test_in, y_test_in, _ = load_split("ids2018_test")
    X_test_cross, y_test_cross, _ = load_split("cicids2017_test")
    X_test_temporal, y_test_temporal, _ = load_split("ids2018_temporal_test")

    scaler = StandardScaler()
    scaler.fit(X_train)  # fitted for completeness; tree models below don't need scaling
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    results = {}

    print("Training RandomForest ...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)  # tree models don't need scaling
    evaluate(rf, X_test_in, y_test_in, "rf_in_distribution", results)
    evaluate(rf, X_test_cross, y_test_cross, "rf_cross_dataset", results)
    if X_test_temporal is not None:
        evaluate(rf, X_test_temporal, y_test_temporal, "rf_temporal_drift", results)
    joblib.dump(rf, MODEL_DIR / "rf.joblib")

    print("\nTraining XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    evaluate(xgb, X_test_in, y_test_in, "xgb_in_distribution", results)
    evaluate(xgb, X_test_cross, y_test_cross, "xgb_cross_dataset", results)
    if X_test_temporal is not None:
        evaluate(xgb, X_test_temporal, y_test_temporal, "xgb_temporal_drift", results)
    joblib.dump(xgb, MODEL_DIR / "xgb.joblib")

    with open(RESULTS_DIR / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved models to {MODEL_DIR}, metrics to {RESULTS_DIR / 'model_metrics.json'}")
    print("\nExpect: *_cross_dataset metrics noticeably lower than *_in_distribution.")
    print("That gap is Gap 1 (cross-dataset generalization failure) — your RQ1/RQ2 evidence.")
    if X_test_temporal is not None:
        print(
            "\n*_temporal_drift metrics isolate TRUE concept drift (same dataset, later "
            "days, same feature schema) from feature-scheme mismatch. If cross_dataset "
            "drops much more than temporal_drift, that tells you how much of the "
            "cross-dataset gap is a feature-alignment artifact vs real behavioral shift "
            "-- worth a sentence in your discussion section either way."
        )


if __name__ == "__main__":
    main()
