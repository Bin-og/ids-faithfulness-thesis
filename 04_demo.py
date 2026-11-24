"""
04_demo.py
----------
Live-demo script for the thesis defense. Does NOT retrain or recompute
anything heavy -- it loads the already-trained model (models/xgb.joblib)
and one real attack flow from each of the three test conditions
(in-distribution, cross-dataset, temporal-drift), then for each one:

  1. Predicts attack/benign + confidence
  2. Shows the top-5 features SHAP says drove that decision (text bar chart)
  3. Runs a live faithfulness check: masks those top features (replaces them
     with the training-set mean) and shows how much confidence drops --
     this is "comprehensiveness" computed on a single example, live.

Takes a few seconds to run. Meant to be run live in front of your
supervisor/panel as proof the pipeline works end to end, not just numbers
in a table.

Run: python scripts/04_demo.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_FILE = "xgb.joblib"
TOP_K = 5          # how many features to show / mask
RANDOM_STATE = 42

CONDITIONS = [
    ("IN-DISTRIBUTION", "ids2018_test",
     "held-out traffic from the SAME day(s) the model trained on"),
    ("CROSS-DATASET", "cicids2017_test",
     "a completely different dataset (CICIDS2017)"),
    ("TEMPORAL-DRIFT", "ids2018_temporal_test",
     "same dataset, same attack family, captured one day later"),
]


def bar(value, max_value, width=30):
    """Tiny text bar chart, no plotting library needed for the live demo."""
    n = int(round(width * (abs(value) / max_value))) if max_value > 0 else 0
    return "#" * max(n, 1 if value != 0 else 0)


def load_condition(name):
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main():
    print("=" * 70)
    print("IDS EXPLANATION-FAITHFULNESS LIVE DEMO")
    print("=" * 70)

    model = joblib.load(MODEL_DIR / MODEL_FILE)
    with open(DATA_DIR / "feature_columns.json") as f:
        feat_cols = json.load(f)

    train_df = pd.read_parquet(DATA_DIR / "ids2018_train.parquet")
    feature_means = train_df[feat_cols].values.astype(float).mean(axis=0)

    explainer = shap.TreeExplainer(model)

    for title, split_name, description in CONDITIONS:
        df = load_condition(split_name)
        print("\n" + "-" * 70)
        print(f"[{title}]  ({description})")
        print("-" * 70)
        if df is None:
            print("  (data not found, skipping)")
            continue

        attacks = df[df["Label"] == 1]
        if len(attacks) == 0:
            print("  (no attack rows found in this split, skipping)")
            continue

        # Pick a flow the model actually calls ATTACK with high confidence.
        # Picking any random attack row risks landing on one the model
        # already misses (predicts BENIGN) -- the faithfulness check is only
        # a meaningful demo on a case the model confidently got right.
        candidates = attacks.sample(min(len(attacks), 3000), random_state=RANDOM_STATE)
        cand_probs = model.predict_proba(candidates[feat_cols].values.astype(float))[:, 1]
        confident_mask = cand_probs >= 0.9
        if confident_mask.sum() > 0:
            pool = candidates[confident_mask]
        elif (cand_probs >= 0.5).sum() > 0:
            pool = candidates[cand_probs >= 0.5]
        else:
            print("  (model does not confidently flag ANY sampled attack here -- "
                  "showing its single highest-confidence attempt instead)")
            pool = candidates.iloc[[np.argmax(cand_probs)]]
        # Detailed single example (for the live walkthrough)
        row = pool.sample(1, random_state=RANDOM_STATE)
        x = row[feat_cols].values.astype(float)

        pred_proba = model.predict_proba(x)[:, 1][0]
        pred_label = "ATTACK" if pred_proba >= 0.5 else "BENIGN"
        true_label = "ATTACK" if row["Label"].values[0] == 1 else "BENIGN"
        attack_type = row["attack_type"].values[0] if "attack_type" in row.columns else "?"

        print(f"  True label:      {true_label}  (attack_type: {attack_type})")
        print(f"  Model prediction: {pred_label}   (confidence={pred_proba:.4f})")

        shap_vals = explainer.shap_values(x)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_row = shap_vals[0]

        order = np.argsort(-np.abs(shap_row))[:TOP_K]
        max_abs = np.abs(shap_row[order]).max()

        print(f"\n  Top {TOP_K} features driving this decision (SHAP):")
        for idx in order:
            sign = "+" if shap_row[idx] >= 0 else "-"
            print(f"    {feat_cols[idx]:<28} {sign}{abs(shap_row[idx]):.4f}  {bar(shap_row[idx], max_abs)}")

        # --- live faithfulness check on THIS one example ------------------
        x_masked = x.copy()
        x_masked[0, order] = feature_means[order]
        masked_proba = model.predict_proba(x_masked)[:, 1][0]
        drop = pred_proba - masked_proba

        print(f"\n  Faithfulness check (this one flow): mask those top {TOP_K} features and re-predict ...")
        print(f"    confidence BEFORE masking: {pred_proba:.4f}")
        print(f"    confidence AFTER masking:  {masked_proba:.4f}")
        print(f"    confidence drop:           {drop:.4f}  "
              f"({'faithful -- those features really mattered' if drop > 0.3 else 'NOT faithful -- confidence barely moved despite SHAP ranking these as top features'})")

        # --- batch average over up to 20 confidently-correct attack flows -
        # A single example can go either way by chance; this is the same
        # measure averaged over more examples, the way the real results
        # (faithfulness_results.csv) were computed, so it's not cherry-picked.
        batch = pool.sample(min(len(pool), 20), random_state=RANDOM_STATE)
        Xb = batch[feat_cols].values.astype(float)
        base_b = model.predict_proba(Xb)[:, 1]
        shap_b = explainer.shap_values(Xb)
        if isinstance(shap_b, list):
            shap_b = shap_b[1]
        Xb_masked = Xb.copy()
        for i in range(len(Xb)):
            top_idx = np.argsort(-np.abs(shap_b[i]))[:TOP_K]
            Xb_masked[i, top_idx] = feature_means[top_idx]
        masked_b = model.predict_proba(Xb_masked)[:, 1]
        mean_drop = float(np.mean(base_b - masked_b))
        print(f"\n  Same check averaged over {len(batch)} confidently-correct attack flows:")
        print(f"    mean confidence drop: {mean_drop:.4f}")

    print("\n" + "=" * 70)
    print("Done. Compare the three blocks above: in-distribution should show")
    print("a large confidence drop after masking (faithful explanation).")
    print("Cross-dataset / temporal-drift should show a much smaller drop --")
    print("same kind of SHAP ranking, but it no longer reflects what's")
    print("actually driving the prediction.")
    print("=" * 70)


if __name__ == "__main__":
    main()
