"""
model_io.py — Save the final aggregated model to disk.

Called by server_app.py once `strategy.start()` returns. Writes three files:
  • prs_global_model.pkl     — sklearn SGDClassifier, ready for predict_proba
  • prs_global_weights.npz   — raw numpy coef + intercept
  • training_history.json    — per-TRE metrics for every round
"""

import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np

from prs_fed.task import create_model, set_parameters


def save_final_artifacts(
    coef            : np.ndarray,
    intercept       : np.ndarray,
    history         : dict,
    n_features      : int,
    n_rounds        : int,
    results_dir     : str,
) -> dict[str, str]:
    """Write all three result files; return a dict of paths."""
    os.makedirs(results_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # Raw numpy weights (framework-agnostic — load with `np.load`)
    npz_path = os.path.join(results_dir, "prs_global_weights.npz")
    np.savez(npz_path, coef=coef, intercept=intercept)
    paths["weights"] = npz_path

    # Pickled sklearn model (predict_proba-ready)
    model = create_model()
    set_parameters(model, [coef, intercept])
    pkl_path = os.path.join(results_dir, "prs_global_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    paths["model"] = pkl_path

    # JSON history + metadata
    meta = {
        "saved_at"       : datetime.now(timezone.utc).isoformat(),
        "model_type"     : "SGDClassifier (logistic regression, balanced)",
        "n_features"     : int(n_features),
        "n_rounds"       : int(n_rounds),
        "deployment_mode": "Flower Message API (SuperLink + SuperNode)",
        "history"        : history,
    }
    json_path = os.path.join(results_dir, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    paths["history"] = json_path

    return paths
