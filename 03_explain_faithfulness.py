"""
03_explain_faithfulness.py
---------------------------
Core novelty script: computes SHAP explanations for the trained model on
BOTH the in-distribution test set and the cross-dataset test set, then
measures explanation faithfulness (comprehensiveness + sufficiency,
ERASER-style) on each. The comparison between the two faithfulness scores
is your RQ3 result.

Definitions used (DeYoung et al., ERASER, 2020):
  - comprehensiveness: how much predicted-class confidence DROPS when the
    top-k most important features (by |SHAP value|) are removed (set to
    the training-set mean / masked). High comprehensiveness = features
    marked "important" really were driving the prediction.
  - sufficiency: how much predicted-class confidence is RETAINED when
    ONLY the top-k features are kept (everything else masked). High
    sufficiency = the top-k features alone are enough to reach the same
    decision.

k is swept over {5, 10, 15, 20}% of features and averaged (comprehensiveness
and sufficiency are normally reported as an AUC over the deletion/insertion
curve, not a single k).

Run: python scripts/03_explain_faithfulness.py
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
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Use XGBoost by default (swap to "rf.joblib" if you prefer RandomForest --
# TreeExplainer supports both, but report whichever you trust more).
MODEL_FILE = "xgb.joblib"
N_SAMPLE = 1000       # subsample test sets for SHAP+faithfulness (full sets are slow)
K_FRACTIONS = [0.05, 0.10, 0.15, 0.20]
RANDOM_STATE = 42


def load_split(name):
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return None, None
    df = pd.read_parquet(path)
    with open(DATA_DIR / "feature_columns.json") as f:
        feat_cols = json.load(f)
    return df, feat_cols


def get_attack_confidence(model, X):
    """Predicted probability of the 'attack' class (index 1)."""
    return model.predict_proba(X)[:, 1]


def rank_stability(shap_values_a, shap_values_b, feat_cols, k=3):
    """
    Rank stability the way the closest prior work (the 2026 Zenodo preprint)
    reports it: aggregate feature importance (mean |SHAP value| across rows)
    computed separately for two conditions (e.g. in-distribution vs shifted),
    then Spearman rank correlation between the two importance rankings.

    rho near 1.0  -> which features matter stays the same under shift
                     (this is what the prior work found: rho=0.997)
    rho well below 1.0 -> the model's own notion of "what matters" reshuffles

    This alone does NOT tell you whether the explanations are still faithful
    (i.e. whether those top features actually drive the prediction) -- that's
    what comprehensiveness/sufficiency below measure. Reporting both is the
    point: you can have high rank stability (same features stay "important")
    while faithfulness still collapses (those features stop actually
    explaining the decision). That combination is your differentiator from
    the prior work.
    """
    from scipy.stats import spearmanr

    importance_a = np.mean(np.abs(shap_values_a), axis=0)
    importance_b = np.mean(np.abs(shap_values_b), axis=0)
    rho, pval = spearmanr(importance_a, importance_b)

    top_k_a = set(np.argsort(-importance_a)[:k])
    top_k_b = set(np.argsort(-importance_b)[:k])
    topk_overlap = len(top_k_a & top_k_b) / k

    return float(rho), float(pval), float(topk_overlap)


def comprehensiveness_sufficiency(model, X, shap_values, feature_means, k_frac):
    """
    X: (n, d) raw feature matrix
    shap_values: (n, d) SHAP values for the SAME rows, same feature order
    feature_means: (d,) values used to "mask" a feature (train-set mean)
    Returns mean comprehensiveness, mean sufficiency for this k_frac.
    """
    n, d = X.shape
    k = max(1, int(round(d * k_frac)))
    base_conf = get_attack_confidence(model, X)

    order = np.argsort(-np.abs(shap_values), axis=1)  # most important first
    top_k_idx = order[:, :k]

    # comprehensiveness: remove (mask) top-k features
    X_comp = X.copy()
    for i in range(n):
        X_comp[i, top_k_idx[i]] = feature_means[top_k_idx[i]]
    comp_conf = get_attack_confidence(model, X_comp)
    comprehensiveness = np.mean(base_conf - comp_conf)

    # sufficiency: keep ONLY top-k features, mask everything else
    X_suff = np.tile(feature_means, (n, 1))
    for i in range(n):
        X_suff[i, top_k_idx[i]] = X[i, top_k_idx[i]]
    suff_conf = get_attack_confidence(model, X_suff)
    sufficiency = np.mean(base_conf - (base_conf - suff_conf))  # = mean(suff_conf), kept explicit for clarity

    return comprehensiveness, np.mean(suff_conf)


def run_for_split(model, explainer, df, feat_cols, feature_means, split_name, rng):
    """
    Computes faithfulness TWO ways for this split:

      scope="all_attacks": every true-attack row, regardless of whether the
        model actually predicted attack for it. This is the headline number
        from before, but it has a confound: for a row the model MISSES
        (predicts benign), the attack-class confidence is already near 0
        before masking anything, so comprehensiveness is trivially ~0 for
        that row -- not because SHAP's explanation is wrong, but because
        there was no confident "attack" decision to explain in the first
        place. A split with low attack recall (e.g. cross-dataset) will
        drag this number down for that reason alone, on top of any real
        unfaithfulness.

      scope="correctly_detected": only rows the model actually predicts as
        ATTACK with >=0.5 confidence. This isolates explanation quality
        from detection failure -- it answers "when the model DOES confidently
        call this an attack, is SHAP's stated reason actually true?",
        independent of how often it misses attacks altogether.

    Report both. If "correctly_detected" still drops substantially under
    shift, that's clean evidence faithfulness degrades independently of
    accuracy. If "correctly_detected" stays high while only "all_attacks"
    collapses, the collapse is mostly an artifact of recall loss, and the
    thesis's faithfulness claim needs to lean on "correctly_detected".
    """
    df_sample = df[df["Label"] == 1]  # true attack rows
    if len(df_sample) > N_SAMPLE:
        df_sample = df_sample.sample(N_SAMPLE, random_state=RANDOM_STATE)
    X_all = df_sample[feat_cols].values.astype(float)

    print(f"\nComputing SHAP values for {split_name} ({len(X_all)} attack rows) ...")
    shap_values_all = explainer.shap_values(X_all)
    if isinstance(shap_values_all, list):  # some SHAP versions return [class0, class1]
        shap_values_all = shap_values_all[1]

    pred_proba_all = get_attack_confidence(model, X_all)
    correct_mask = pred_proba_all >= 0.5
    n_correct = int(correct_mask.sum())
    print(f"  of these, model predicts ATTACK (correctly detected) for {n_correct}/{len(X_all)}")

    rows = []
    for scope, mask in [("all_attacks", np.ones(len(X_all), dtype=bool)),
                         ("correctly_detected", correct_mask)]:
        if mask.sum() == 0:
            print(f"  [{scope}] no rows in this scope -- skipping")
            continue
        X = X_all[mask]
        shap_values = shap_values_all[mask]
        for k_frac in K_FRACTIONS:
            comp, suff = comprehensiveness_sufficiency(model, X, shap_values, feature_means, k_frac)
            rows.append({
                "split": split_name, "scope": scope, "n_rows": int(mask.sum()),
                "k_fraction": k_frac, "comprehensiveness": comp, "sufficiency": suff,
            })
            print(f"  [{scope}] k={int(k_frac*100)}%  comprehensiveness={comp:.4f}  sufficiency={suff:.4f}")

    # rank stability (below) uses the full attack sample, same as before
    return rows, shap_values_all, X_all


def main():
    model = joblib.load(MODEL_DIR / MODEL_FILE)
    rng = np.random.default_rng(RANDOM_STATE)

    train_df, feat_cols = load_split("ids2018_train")
    feature_means = train_df[feat_cols].values.astype(float).mean(axis=0)

    explainer = shap.TreeExplainer(model)

    test_in_df, _ = load_split("ids2018_test")
    test_cross_df, _ = load_split("cicids2017_test")
    test_temporal_df, _ = load_split("ids2018_temporal_test")

    splits = {"in_distribution": test_in_df, "cross_dataset": test_cross_df}
    if test_temporal_df is not None:
        splits["temporal_drift"] = test_temporal_df

    all_rows = []
    shap_by_split = {}
    for name, df in splits.items():
        rows, shap_vals, _ = run_for_split(model, explainer, df, feat_cols, feature_means, name, rng)
        all_rows.extend(rows)
        shap_by_split[name] = shap_vals

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_DIR / "faithfulness_results.csv", index=False)

    # summary comparison at k=10% (report the full sweep in the thesis; this is a quick readout)
    summary = results_df[results_df["k_fraction"] == 0.10].set_index(["split", "scope"])
    print("\n=== Faithfulness summary at k=10% (both scopes) ===")
    print(summary[["n_rows", "comprehensiveness", "sufficiency"]])
    print(
        "\n'all_attacks' includes attacks the model MISSED (predicted benign) -- for those,\n"
        "attack-class confidence is already ~0 before masking, so a low score there can be\n"
        "detection failure, not explanation failure.\n"
        "'correctly_detected' is restricted to attacks the model actually flagged as attack --\n"
        "THIS is the clean test of whether faithfulness degrades independently of accuracy.\n"
        "If 'correctly_detected' comprehensiveness drops substantially from in_distribution to\n"
        "cross_dataset/temporal_drift, that is your RQ3 finding. If it stays high while only\n"
        "'all_attacks' collapses, the collapse is mostly a recall artifact -- say so plainly\n"
        "in the thesis rather than overclaiming independence from accuracy."
    )

    # --- rank stability: in_distribution vs each shifted condition -----
    rank_rows = []
    baseline = shap_by_split["in_distribution"]
    for name, shap_vals in shap_by_split.items():
        if name == "in_distribution":
            continue
        rho, pval, topk_overlap = rank_stability(baseline, shap_vals, feat_cols, k=3)
        rank_rows.append({
            "comparison": f"in_distribution_vs_{name}",
            "spearman_rho": rho,
            "p_value": pval,
            "top3_overlap": topk_overlap,
        })
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(RESULTS_DIR / "rank_stability_results.csv", index=False)

    print("\n=== Rank stability (aggregate feature importance, Spearman rho) ===")
    print(rank_df)
    print(
        "\nHigh rho (features that matter stay the same) alongside a large faithfulness "
        "drop is the key combined finding: the model's explanations LOOK stable by rank, "
        "but stop being faithful. That decoupling is what the closest prior work "
        "(rank-stability only, rho=0.997) did not test for."
    )

    print(f"\nSaved: {RESULTS_DIR / 'faithfulness_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'rank_stability_results.csv'}")


if __name__ == "__main__":
    main()
