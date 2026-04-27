"""
data_utils.py
-------------
Loads and preprocesses the three genetics CSV datasets.

Target: 'case' (binary 0/1 - disease case/control)
Features: 313 SNP dosage columns (0/1/2)

The 'prs' column is used ONLY to recover the true SNP effect sizes
(betas) for the coefficient recovery analysis in Figure 3.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

META_COLS  = ["id", "ageOfEntry", "ageOfExit", "sex", "prs", "case", "ageOfOnset"]
TARGET_COL = "case"


def get_snp_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if ":" in c]


def recover_true_betas(reference_csv: str) -> np.ndarray:
    """
    Recover the true PRS effect sizes (betas) by regressing prs on SNPs
    using the normal/reference dataset. These are the ground-truth weights
    that the federated model's coefficients are compared against (Figure 3).
    """
    df       = pd.read_csv(reference_csv)
    snp_cols = get_snp_columns(df)
    X        = df[snp_cols].values.astype(np.float64)
    y        = df["prs"].values.astype(np.float64)
    # Near-zero alpha: PRS is an exact linear combo of SNPs
    betas    = Ridge(alpha=1e-10).fit(X, y).coef_
    return betas


def load_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load a single cohort CSV.

    Features : 313 SNP dosages (standardised)
    Target   : case (0/1)

    Returns X_train, X_test, y_train, y_test, scaler, meta
    """
    df       = pd.read_csv(csv_path)
    snp_cols = get_snp_columns(df)

    X = df[snp_cols].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(int)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    # Deterministic 80/20 split (no shuffle - preserves any ordering)
    split    = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    pct_under41 = float((df["ageOfEntry"] < 41).mean() * 100)
    pct_over69  = float((df["ageOfEntry"] > 69).mean() * 100)

    meta = dict(
        cohort_label  = os.path.splitext(os.path.basename(csv_path))[0],
        n_samples     = len(df),
        n_features    = X_train.shape[1],
        n_snps        = len(snp_cols),
        snp_cols      = snp_cols,
        age_mean      = float(df["ageOfEntry"].mean()),
        age_std       = float(df["ageOfEntry"].std()),
        pct_under41   = pct_under41,
        pct_over69    = pct_over69,
        n_cases       = int(y.sum()),
        n_controls    = int((y == 0).sum()),
        case_rate     = float(y.mean()),
        n_train       = len(y_train),
        n_test        = len(y_test),
        cases_train   = int(y_train.sum()),
        cases_test    = int(y_test.sum()),
    )

    return X_train, X_test, y_train, y_test, scaler, meta


def load_all_datasets(data_dir: str = "."):
    """Load all three cohort CSVs in A/B/C order (young/old/normal)."""
    files = {
        "young" : os.path.join(data_dir, "USA_young.csv"),
        "old"   : os.path.join(data_dir, "USA_old.csv"),
        "normal": os.path.join(data_dir, "USA_normal.csv"),
    }
    datasets = []
    for label, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        print(f"  Loading {label} cohort …")
        datasets.append(load_dataset(path))
    return datasets
