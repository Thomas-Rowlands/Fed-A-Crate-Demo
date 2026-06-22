"""
model_io.py — Save and load the trained federated logistic regression model.
"""

import os, json, pickle
import numpy as np
from datetime import datetime, timezone
from scipy.stats import pearsonr


def save_model(model, feature_cols, cohort_metas, n_rounds,
               output_dir="results", true_betas=None):
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    pkl_path = os.path.join(output_dir, "prs_global_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    paths["pickle"] = pkl_path

    npz_path = os.path.join(output_dir, "prs_global_weights.npz")
    save_kwargs = dict(coef=model.coef_[0], intercept=model.intercept_)
    if true_betas is not None:
        save_kwargs["true_betas"] = true_betas
    np.savez(npz_path, **save_kwargs)
    paths["numpy"] = npz_path if npz_path.endswith(".npz") else npz_path + ".npz"

    coef_stats = {}
    if true_betas is not None:
        r, p = pearsonr(true_betas, model.coef_[0])
        rmse = float(np.sqrt(np.mean((true_betas - model.coef_[0])**2)))
        mask = np.abs(true_betas) > 1e-8
        mape = float(np.mean(np.abs(
            (true_betas[mask] - model.coef_[0][mask]) / true_betas[mask])) * 100)
        coef_stats = {"pearson_r": round(float(r), 4), "pearson_p": float(p),
                      "rmse": rmse, "mape_pct": mape}

    meta = {
        "saved_at"      : datetime.now(timezone.utc).isoformat(),
        "model_type"    : "SGDClassifier (logistic regression, class_weight=balanced)",
        "target"        : "case (0=control, 1=disease case)",
        "n_features"    : len(feature_cols),
        "feature_names" : feature_cols,
        "federation"    : {"framework": "Flower (flwr)", "strategy": "FedAvg",
                           "n_rounds": n_rounds, "local_epochs": 5,
                           "n_clients": 3},
        "cohorts"       : [
            {"label": m["cohort_label"], "n_samples": m["n_samples"],
             "case_rate": round(m["case_rate"], 4),
             "pct_under41": round(m.get("pct_under41", 0), 1),
             "pct_over69":  round(m.get("pct_over69",  0), 1)}
            for m in cohort_metas
        ] if cohort_metas else [],
        "coefficient_recovery": coef_stats,
        "usage_example": (
            "from model_io import load_model\n"
            "model, meta = load_model('results/')\n"
            "probs = model.predict_proba(X_scaled)[:, 1]\n"
            "preds = (probs >= 0.5).astype(int)"
        ),
    }
    json_path = os.path.join(output_dir, "prs_model_metadata.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    paths["metadata"] = json_path
    return paths


def load_model(output_dir="results"):
    pkl_path  = os.path.join(output_dir, "prs_global_model.pkl")
    json_path = os.path.join(output_dir, "prs_model_metadata.json")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Model not found at {pkl_path}.")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    meta = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
    return model, meta


def load_weights(output_dir="results"):
    npz_path = os.path.join(output_dir, "prs_global_weights.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Weights not found at {npz_path}.")
    w = np.load(npz_path)
    return w["coef"], w["intercept"]
