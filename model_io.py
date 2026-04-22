"""
model_io.py
-----------
Save and load the final trained federated PRS model.

Two formats are written:
  1. prs_global_model.pkl   – full sklearn SGDRegressor pickle (easiest to reuse)
  2. prs_global_weights.npz – raw numpy arrays (coef_ + intercept_) for
                               framework-agnostic use / inspection

A human-readable JSON metadata sidecar is also written:
  3. prs_model_metadata.json – feature list, training summary, cohort stats

Usage (loading the saved model):
---------------------------------
    from model_io import load_model
    model, meta = load_model("results/")
    predictions = model.predict(X_new)   # X_new must be scaled the same way

Or using raw numpy weights:
    import numpy as np
    w = np.load("results/prs_global_weights.npz")
    predictions = X_new_scaled @ w["coef"] + w["intercept"]
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from sklearn.linear_model import SGDRegressor


def save_model(
    model       : SGDRegressor,
    feature_cols: list[str],
    cohort_metas: list[dict],
    n_rounds    : int,
    output_dir  : str = "results",
) -> dict[str, str]:
    """
    Persist the final federated model to disk in multiple formats.

    Parameters
    ----------
    model        : fitted SGDRegressor with coef_ and intercept_ set
    feature_cols : ordered list of feature names (SNP IDs + 'ageOfEntry')
    cohort_metas : list of per-cohort metadata dicts from data_utils
    n_rounds     : number of federation rounds completed
    output_dir   : directory to write into

    Returns
    -------
    dict mapping format name → file path
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # ── 1. Pickle (full sklearn model) ───────────────────────────────────────
    pkl_path = os.path.join(output_dir, "prs_global_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    paths["pickle"] = pkl_path
    print(f"  [save] Pickle  → {pkl_path}")

    # ── 2. Numpy weights (framework-agnostic) ────────────────────────────────
    npz_path = os.path.join(output_dir, "prs_global_weights.npz")
    np.savez(
        npz_path,
        coef      = model.coef_,
        intercept = model.intercept_,
    )
    paths["numpy"] = npz_path + ".npz" if not npz_path.endswith(".npz") else npz_path
    # np.savez always appends .npz
    actual_npz = npz_path if npz_path.endswith(".npz") else npz_path + ".npz"
    paths["numpy"] = actual_npz
    print(f"  [save] Weights → {actual_npz}")

    # ── 3. JSON metadata sidecar ─────────────────────────────────────────────
    meta = {
        "saved_at"        : datetime.utcnow().isoformat() + "Z",
        "model_type"      : "SGDRegressor (linear regression)",
        "target"          : "prs (Polygenic Risk Score, continuous)",
        "n_features"      : len(feature_cols),
        "n_snp_features"  : len(feature_cols) - 1,   # exclude ageOfEntry
        "feature_names"   : feature_cols,
        "federation"      : {
            "framework"      : "Flower (flwr)",
            "strategy"       : "FedAvg (weighted by n_samples)",
            "n_rounds"       : n_rounds,
            "local_epochs"   : 5,
            "n_clients"      : len(cohort_metas),
        },
        "cohorts"         : [
            {
                "label"     : m["cohort_label"],
                "n_samples" : m["n_samples"],
                "age_mean"  : round(m["age_mean"], 2),
                "age_std"   : round(m["age_std"],  2),
                "prs_mean"  : round(m["prs_mean"], 4),
                "case_rate" : round(m["case_rate"], 4),
            }
            for m in cohort_metas
        ],
        "model_params"    : {
            "coef_norm"      : float(np.linalg.norm(model.coef_)),
            "intercept"      : float(model.intercept_[0]),
            "n_nonzero_coef" : int(np.sum(model.coef_ != 0)),
        },
        "preprocessing"   : (
            "Features standardised per cohort using sklearn StandardScaler "
            "(zero mean, unit variance). Apply the same scaler before predicting."
        ),
        "usage_example"   : (
            "from model_io import load_model\n"
            "model, meta = load_model('results/')\n"
            "# X_new must be scaled with the same StandardScaler used during training\n"
            "predictions = model.predict(X_new_scaled)"
        ),
    }

    json_path = os.path.join(output_dir, "prs_model_metadata.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    paths["metadata"] = json_path
    print(f"  [save] Metadata→ {json_path}")

    return paths


def load_model(output_dir: str = "results") -> tuple[SGDRegressor, dict]:
    """
    Load the saved federated model from output_dir.

    Returns
    -------
    model : sklearn SGDRegressor ready for .predict()
    meta  : dict parsed from prs_model_metadata.json
    """
    pkl_path  = os.path.join(output_dir, "prs_global_model.pkl")
    json_path = os.path.join(output_dir, "prs_model_metadata.json")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Model pickle not found at {pkl_path}. "
            "Run main.py first to train and save the model."
        )

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    meta = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)

    return model, meta


def load_weights(output_dir: str = "results") -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw numpy weights (no sklearn dependency).

    Returns
    -------
    coef      : shape (n_features,)
    intercept : shape (1,)
    """
    npz_path = os.path.join(output_dir, "prs_global_weights.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Weights file not found at {npz_path}.")
    w = np.load(npz_path)
    return w["coef"], w["intercept"]
