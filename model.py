"""
model.py
--------
Defines the linear regression model used for polygenic risk score prediction
and helper functions to convert between sklearn model parameters and the
numpy arrays that Flower exchanges between server and clients.
"""

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ── Model factory ───────────────────────────────────────────────────────────

def create_model(n_features: int, random_state: int = 42) -> SGDRegressor:
    """
    Return an SGDRegressor configured for online / incremental learning.
    Using SGDRegressor (rather than LinearRegression) lets each client train
    for multiple local epochs by calling partial_fit, which mirrors how neural
    network clients work in Flower and makes the FedAvg aggregation natural.
    """
    return SGDRegressor(
        loss           = "squared_error",
        penalty        = "l2",
        alpha          = 1e-4,          # L2 regularisation – helps with 313 SNPs
        learning_rate  = "invscaling",
        eta0           = 0.01,
        max_iter       = 1,             # we control epochs manually via partial_fit
        warm_start     = True,
        random_state   = random_state,
    )


# ── Parameter serialisation ─────────────────────────────────────────────────

def get_parameters(model: SGDRegressor) -> list[np.ndarray]:
    """
    Extract the model weights and bias as a list of numpy arrays.
    If the model has never been fit, return zero-initialised arrays.
    """
    if hasattr(model, "coef_"):
        return [model.coef_.copy(), np.array([model.intercept_[0]])]
    # Not yet fit – return zeros (shape will be inferred later)
    return []


def set_parameters(model: SGDRegressor, parameters: list[np.ndarray]) -> SGDRegressor:
    """
    Push aggregated parameters back into a model instance.
    Handles the case where the model has not yet been initialised by sklearn.
    """
    coef, intercept = parameters
    model.coef_      = coef.copy()
    model.intercept_ = intercept.copy()
    return model


# ── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate_model(model: SGDRegressor, X: np.ndarray, y: np.ndarray) -> dict:
    """Return MSE and R² for a fitted model on the given data."""
    y_pred = model.predict(X)
    mse    = float(mean_squared_error(y, y_pred))
    r2     = float(r2_score(y, y_pred))
    return {"mse": mse, "r2": r2}
