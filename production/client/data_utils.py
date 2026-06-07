"""
data_utils.py — Production version.

Each TRE loads ONLY its own cohort CSV. There is no global view of the
other cohorts and no recovery of "true betas" (that requires the PRS
column across the reference cohort — impossible without data sharing).

Target  : case (binary 0/1)
Features: 313 SNP dosages (standardised locally within this TRE)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COL = "case"


def get_snp_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if ":" in c]


def load_local_cohort(csv_path: str, test_size: float = 0.2):
    """
    Load this TRE's cohort CSV and prepare train/test splits.

    Standardisation is fit on local training data only — the scaler's
    parameters never leave this TRE.
    """
    df       = pd.read_csv(csv_path)
    snp_cols = get_snp_columns(df)

    X = df[snp_cols].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(int)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    split   = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    meta = dict(
        cohort_label = os.path.splitext(os.path.basename(csv_path))[0],
        n_samples    = len(df),
        n_features   = X_train.shape[1],
        snp_cols     = snp_cols,
        age_mean     = float(df["ageOfEntry"].mean()),
        pct_under41  = float((df["ageOfEntry"] < 41).mean() * 100),
        pct_over69   = float((df["ageOfEntry"] > 69).mean() * 100),
        n_cases      = int(y.sum()),
        n_controls   = int((y == 0).sum()),
        case_rate    = float(y.mean()),
        n_train      = len(y_train),
        n_test       = len(y_test),
        cases_train  = int(y_train.sum()),
        cases_test   = int(y_test.sum()),
    )
    return X_train, X_test, y_train, y_test, scaler, meta
