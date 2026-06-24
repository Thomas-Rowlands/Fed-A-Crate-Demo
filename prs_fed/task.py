"""
task.py — Pure ML logic for PRS case/control classification.

This module contains everything that does NOT depend on Flower:
  • SGDClassifier creation and parameter (de)serialisation
  • Local CSV loading and standardisation
  • Metric computation

It can be imported from both server_app.py and client_app.py; the server
doesn't actually train, but it uses `create_model()` to materialise the
final aggregated weights for saving.

Nothing in this file changed from the previous (NumPyClient-based) version
except for the consolidation — model.py and data_utils.py are now one module
to match the Flower App convention.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model    import SGDClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (accuracy_score, roc_auc_score, log_loss,
                                      precision_score, recall_score, f1_score)
from scipy.special           import expit


# ── Model factory ────────────────────────────────────────────────────────────

def create_model(random_state: int = 42) -> SGDClassifier:
    """Logistic regression with class_weight='balanced' (~1-2% case prevalence)."""
    return SGDClassifier(
        loss         = "log_loss",
        penalty      = "l2",
        alpha        = 1e-4,
        learning_rate= "invscaling",
        eta0         = 0.01,
        max_iter     = 1,
        warm_start   = True,
        class_weight = "balanced",
        random_state = random_state,
    )


# ── Parameter (de)serialisation ──────────────────────────────────────────────
# Under the Message API the ArrayRecord wraps these numpy arrays.
# We export plain numpy here; the (de)serialisation to/from ArrayRecord
# happens at the app boundary (server_app.py / client_app.py).

def get_parameters(model: SGDClassifier) -> list[np.ndarray]:
    """Return [coef (1-D), intercept (1-D)]."""
    if hasattr(model, "coef_"):
        return [model.coef_[0].copy(), model.intercept_.copy()]
    return []


def set_parameters(model: SGDClassifier, parameters: list[np.ndarray]) -> SGDClassifier:
    """Push global parameters into a fresh model so it can `partial_fit` from them."""
    coef, intercept = parameters
    model.coef_      = coef.reshape(1, -1).copy()
    model.intercept_ = intercept.copy()
    if not hasattr(model, "classes_"):
        model.classes_ = np.array([0, 1])
    return model


# ── Local-cohort data loading ────────────────────────────────────────────────

TARGET_COL = "case"


def get_snp_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if ":" in c]


def load_local_cohort(csv_path: str, test_size: float = 0.2):
    """
    Load a single TRE's cohort CSV. Standardisation is fit on local
    training data only; the scaler's parameters never leave this process.
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
        n_cases      = int(y.sum()),
        n_controls   = int((y == 0).sum()),
        case_rate    = float(y.mean()),
        n_train      = len(y_train),
        n_test       = len(y_test),
    )
    return X_train, X_test, y_train, y_test, scaler, meta


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model: SGDClassifier, X: np.ndarray, y: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """Macro-averaged classification metrics (chosen to handle class imbalance)."""
    coef      = model.coef_[0]
    intercept = model.intercept_[0]
    probs     = expit(X @ coef + intercept)
    preds     = (probs >= threshold).astype(int)

    if len(np.unique(y)) < 2:
        return dict(accuracy=float("nan"), auc=float("nan"),
                    log_loss_val=float("nan"), precision=float("nan"),
                    recall=float("nan"), f1=float("nan"))

    return dict(
        accuracy    = float(accuracy_score(y, preds)),
        auc         = float(roc_auc_score(y, probs)),
        log_loss_val= float(log_loss(y, probs)),
        precision   = float(precision_score(y, preds, average="macro", zero_division=0)),
        recall      = float(recall_score(y,    preds, average="macro", zero_division=0)),
        f1          = float(f1_score(y,        preds, average="macro", zero_division=0)),
    )
