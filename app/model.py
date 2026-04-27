"""
model.py
--------
Logistic regression model for federated case/control PRS classification.

Uses SGDClassifier with log_loss (equivalent to logistic regression) and
class_weight='balanced' to handle the severe class imbalance (~1-2% cases).
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    precision_score, recall_score, f1_score,
)
from scipy.special import expit


def create_model(random_state: int = 42) -> SGDClassifier:
    """
    SGDClassifier configured for federated logistic regression.
    class_weight='balanced' is critical: with ~1% cases, an unweighted
    model trivially predicts all-negative.
    """
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


def get_parameters(model: SGDClassifier) -> list:
    if hasattr(model, "coef_"):
        return [model.coef_[0].copy(), model.intercept_.copy()]
    return []


def set_parameters(model, parameters, n_features=None):
    coef, intercept = parameters
    model.coef_      = coef.reshape(1, -1).copy()
    model.intercept_ = intercept.copy()
    if not hasattr(model, "classes_"):
        model.classes_ = np.array([0, 1])
    return model


def evaluate_model(model, X, y, threshold=0.5):
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
        precision   = float(precision_score(y, preds, zero_division=0)),
        recall      = float(recall_score(y, preds, zero_division=0)),
        f1          = float(f1_score(y, preds, zero_division=0)),
    )
