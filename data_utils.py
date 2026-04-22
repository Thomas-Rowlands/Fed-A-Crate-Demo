"""
data_utils.py
-------------
Loads and preprocesses the three genetics CSV datasets for federated PRS learning.
Each dataset represents an age-imbalanced cohort (young / normal / old).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Column definitions ──────────────────────────────────────────────────────
META_COLS   = ["id", "ageOfEntry", "ageOfExit", "sex", "case", "ageOfOnset"]
TARGET_COL  = "prs"

# Age feature we include as a predictor alongside SNPs
AGE_FEATURE = "ageOfEntry"


def get_snp_columns(df: pd.DataFrame) -> list[str]:
    """Return every column that looks like a SNP locus (contains ':')."""
    return [c for c in df.columns if ":" in c]


def load_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load a single CSV, extract features (SNPs + age) and target (prs).

    Returns
    -------
    X_train, X_test, y_train, y_test  – numpy arrays (float32)
    scaler                             – fitted StandardScaler (for reporting)
    meta                               – dict with cohort summary statistics
    """
    df = pd.read_csv(csv_path)

    snp_cols = get_snp_columns(df)

    # Feature matrix: SNP dosages (0/1/2) + age of entry
    feature_cols = snp_cols + [AGE_FEATURE]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    meta = {
        "n_samples"     : len(df),
        "n_features"    : X.shape[1],
        "n_snps"        : len(snp_cols),
        "age_mean"      : float(df[AGE_FEATURE].mean()),
        "age_std"       : float(df[AGE_FEATURE].std()),
        "age_min"       : float(df[AGE_FEATURE].min()),
        "age_max"       : float(df[AGE_FEATURE].max()),
        "prs_mean"      : float(df[TARGET_COL].mean()),
        "prs_std"       : float(df[TARGET_COL].std()),
        "case_rate"     : float(df["case"].mean()),
        "cohort_label"  : os.path.splitext(os.path.basename(csv_path))[0],
    }

    return X_train, X_test, y_train, y_test, scaler, meta


def load_all_datasets(data_dir: str = "."):
    """
    Load all three cohort CSVs.  Returns a list of (X_train, X_test, y_train,
    y_test, scaler, meta) tuples – one per client.
    """
    files = {
        "young" : os.path.join(data_dir, "USA_young.csv"),
        "normal": os.path.join(data_dir, "USA_normal.csv"),
        "old"   : os.path.join(data_dir, "USA_old.csv"),
    }

    datasets = []
    for label, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        print(f"  Loading {label} cohort from {path} …")
        result = load_dataset(path)
        datasets.append(result)

    return datasets
