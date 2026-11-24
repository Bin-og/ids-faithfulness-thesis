"""
01_preprocess.py
-----------------
Loads the raw CSE-CIC-IDS2018 and CICIDS2017 CSV files, cleans them, aligns
both datasets to a common feature schema, and writes train/test splits.

Expected input layout (adjust RAW_DIR paths below if yours differs):
  data/raw/ids2018/*.csv       <- all CSE-CIC-IDS2018 daily CSVs
  data/raw/cicids2017/*.csv    <- all CICIDS2017 daily CSVs

Output:
  data/processed/ids2018_train.parquet
  data/processed/ids2018_test.parquet
  data/processed/cicids2017_test.parquet   (used ONLY for cross-dataset eval)
  data/processed/feature_columns.json      (the shared feature set used everywhere)

Run: python scripts/01_preprocess.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL_CANDIDATES = ["Label", " Label", "label"]
BENIGN_TOKENS = {"benign", "normal"}


def clean_colname(c: str) -> str:
    """Both datasets ship with inconsistent whitespace/casing in headers."""
    c = c.strip()
    c = re.sub(r"\s+", " ", c)
    return c


def load_folder(folder: Path, files=None) -> pd.DataFrame:
    """If `files` is given, load only those (used to build the temporal split);
    otherwise load every CSV in the folder."""
    csvs = files if files is not None else sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in {folder}. Download the dataset first "
            f"(see README.md) and place the daily CSVs there."
        )
    frames = []
    for f in csvs:
        print(f"  loading {f.name} ...")
        df = pd.read_csv(f, low_memory=False, encoding="latin1")
        df.columns = [clean_colname(c) for c in df.columns]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    print(f"  -> {len(full):,} rows from {len(csvs)} file(s)")
    return full


# CSE-CIC-IDS2018 ships one CSV per capture day, and a different day usually
# means a DIFFERENT attack campaign (e.g. Botnet on 02-03, SSH/FTP brute-force
# on 02-14). Alphabetically sorting filenames does NOT track calendar order
# ("Friday-..." sorts before "Wednesday-..."), and even a correct date sort
# would still pair up days with unrelated attack types, which tests
# unseen-attack-type generalization, not time-based drift.
#
# For a genuine temporal-drift test, hold out a later day that shares its
# attack category with an earlier day already in the training set. With the
# default download set (Thursday-22-02-2018 and Friday-23-02-2018 both carry
# Brute-Force-Web/XSS + SQL Injection, one day apart), holding out the Friday
# file isolates real day-to-day drift. Add/replace filenames here if you
# download a different pair.
TEMPORAL_TEST_FILENAME_PREFIXES = ("Friday-23-02-2018",)


def split_files_for_temporal_arm(folder: Path):
    all_files = sorted(folder.glob("*.csv"))
    if len(all_files) < 2:
        print(
            "  *** Only 1 file found -> can't build a temporal-drift arm "
            "(need at least 2 daily captures). Skipping temporal split."
        )
        return None, None
    late_days = [f for f in all_files if f.name.startswith(TEMPORAL_TEST_FILENAME_PREFIXES)]
    early_days = [f for f in all_files if f not in late_days]
    if not late_days:
        print(
            f"  *** None of the downloaded files match "
            f"{TEMPORAL_TEST_FILENAME_PREFIXES} -> skipping temporal split.\n"
            f"  *** Files found: {[f.name for f in all_files]}"
        )
        return None, None
    if not early_days:
        print("  *** All files are temporal-test files, nothing left to train on -> skipping.")
        return None, None
    print(f"  temporal split: train/in-dist on {[f.name for f in early_days]}")
    print(f"                  temporal-drift test on {[f.name for f in late_days]}")
    return early_days, late_days


def find_label_col(df: pd.DataFrame) -> str:
    for cand in LABEL_COL_CANDIDATES:
        if cand.strip() in df.columns:
            return cand.strip()
    raise KeyError(
        f"Could not find a label column. Columns present: {list(df.columns)[:10]}..."
    )


def binarize_label(series: pd.Series) -> pd.Series:
    """Binary attack/benign target. (Multi-class label is kept separately
    as 'attack_type' in case you want per-attack-type breakdowns later.)"""
    norm = series.astype(str).str.strip().str.lower()
    return (~norm.isin(BENIGN_TOKENS)).astype(int)


def basic_clean(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.rename(columns={label_col: "Label"})
    # numeric feature columns only
    feat_cols = [c for c in df.columns if c != "Label"]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=feat_cols, how="any")
    print(f"  dropped {before - len(df):,} rows with NaN/Inf")
    df["attack_type"] = df["Label"].astype(str).str.strip()
    df["Label"] = binarize_label(df["Label"])
    return df


def main():
    print("Splitting CSE-CIC-IDS2018 files into early/late days for the temporal-drift arm ...")
    early_files, late_files = split_files_for_temporal_arm(RAW_DIR / "ids2018")

    print("\nLoading CSE-CIC-IDS2018 (primary / train+in-distribution-test) ...")
    ids2018 = load_folder(RAW_DIR / "ids2018", files=early_files)
    label_2018 = find_label_col(ids2018)

    ids2018_temporal = None
    if late_files:
        print("\nLoading later-day IDS2018 files (temporal-drift test) ...")
        ids2018_temporal = load_folder(RAW_DIR / "ids2018", files=late_files)

    print("\nLoading CICIDS2017 (cross-dataset test only) ...")
    cicids2017 = load_folder(RAW_DIR / "cicids2017")
    label_2017 = find_label_col(cicids2017)

    # --- align to a shared feature schema FIRST, before any numeric -----
    # cleaning/dropna. IDS2018 and CICIDS2017 each carry extra columns the
    # other doesn't (Timestamp, Flow ID, IP addresses, ...). Running
    # pd.to_numeric + dropna across those extra columns (as the old version
    # of this script did) turns them entirely to NaN and wipes out every
    # row. Restricting to the shared numeric columns BEFORE cleaning avoids
    # that.
    ignore_2018 = {label_2018}
    ignore_2017 = {label_2017}
    cols_2018 = set(ids2018.columns) - ignore_2018
    cols_2017 = set(cicids2017.columns) - ignore_2017
    shared = sorted(cols_2018 & cols_2017)

    only_2018 = sorted(cols_2018 - cols_2017)
    only_2017 = sorted(cols_2017 - cols_2018)
    print(f"\nShared features: {len(shared)}")
    if only_2018:
        print(f"Dropped (IDS2018-only, {len(only_2018)}): {only_2018[:15]}"
              + (" ..." if len(only_2018) > 15 else ""))
    if only_2017:
        print(f"Dropped (CICIDS2017-only, {len(only_2017)}): {only_2017[:15]}"
              + (" ..." if len(only_2017) > 15 else ""))
    if len(shared) < 20:
        print(
            "\n*** WARNING: fewer than 20 shared features survived alignment.\n"
            "*** This usually means column names differ more than whitespace/case\n"
            "*** (e.g. 'Fwd Pkt Len Mean' vs 'Fwd Packet Length Mean'). Open both\n"
            "*** raw headers and add a rename map near the top of this script.\n"
        )

    keep_2018 = shared + [label_2018]
    keep_2017 = shared + [label_2017]
    ids2018 = ids2018[keep_2018]
    cicids2017 = cicids2017[keep_2017]
    if ids2018_temporal is not None:
        ids2018_temporal = ids2018_temporal[keep_2018]

    # --- now clean (numeric coercion + dropna), only on the shared cols -
    ids2018 = basic_clean(ids2018, label_2018)
    cicids2017 = basic_clean(cicids2017, label_2017)
    if ids2018_temporal is not None:
        ids2018_temporal = basic_clean(ids2018_temporal, label_2018)

    # --- split IDS2018 into train / in-distribution test ---------------
    train_df, test_df = train_test_split(
        ids2018, test_size=0.2, random_state=42, stratify=ids2018["Label"]
    )

    print(f"\nFinal shapes:")
    print(f"  train (IDS2018, early days):        {train_df.shape}")
    print(f"  in-distribution test (IDS2018):     {test_df.shape}")
    print(f"  cross-dataset test (CICIDS2017):    {cicids2017.shape}")
    if ids2018_temporal is not None:
        print(f"  temporal-drift test (IDS2018, later days): {ids2018_temporal.shape}")

    train_df.to_parquet(OUT_DIR / "ids2018_train.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "ids2018_test.parquet", index=False)
    cicids2017.to_parquet(OUT_DIR / "cicids2017_test.parquet", index=False)
    if ids2018_temporal is not None:
        ids2018_temporal.to_parquet(OUT_DIR / "ids2018_temporal_test.parquet", index=False)
    with open(OUT_DIR / "feature_columns.json", "w") as f:
        json.dump(shared, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
