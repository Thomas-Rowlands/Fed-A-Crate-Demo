"""
strategy.py — Production FedAvg strategy.

Logs server-side narrative ("broadcasting / aggregating / evaluating")
and saves the final global model + per-round metrics to /results.
"""

import json
import os
import pickle
import numpy as np
from datetime import datetime, timezone

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from server.tre_logger import (
    server_round_header, server_round_footer,
    server_broadcasting, server_received_update, server_eval_summary,
)


class TREStrategy(FedAvg):
    """FedAvg with narrative server-side logging and history tracking."""

    def __init__(self, n_features, n_rounds, results_dir="/results", **kwargs):
        initial_params = ndarrays_to_parameters([
            np.zeros(n_features, dtype=np.float64),
            np.zeros(1,          dtype=np.float64),
        ])
        super().__init__(initial_parameters=initial_params, **kwargs)
        self.n_features   = n_features
        self.n_rounds     = n_rounds
        self.results_dir  = results_dir
        self.history = {"train": {}, "eval": {}, "global_ll": {}, "global": {}}
        self._last_params   = None
        self._last_combined = None

    # ── Configure fit ────────────────────────────────────────────────────────
    def configure_fit(self, server_round, parameters, client_manager):
        server_round_header(server_round, self.n_rounds)
        n_available = client_manager.num_available()
        server_broadcasting(n_available)
        return super().configure_fit(server_round, parameters, client_manager)

    # ── Aggregate fit ────────────────────────────────────────────────────────
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights = [(r.num_examples, parameters_to_ndarrays(r.parameters))
                   for _, r in results]
        total      = sum(n for n, _ in weights)
        agg_coef   = sum(n * p[0] for n, p in weights) / total
        agg_int    = sum(n * p[1] for n, p in weights) / total
        aggregated = ndarrays_to_parameters([agg_coef, agg_int])

        self._last_params = [agg_coef, agg_int]
        server_received_update(len(results))

        # Track per-TRE training metrics for the history file
        per_tre = {}
        for _, fr in results:
            tre_num = fr.metrics.get("tre_num", 0)
            per_tre[tre_num] = {
                "cohort"   : fr.metrics.get("cohort"),
                "train_auc": fr.metrics.get("train_auc"),
                "train_f1" : fr.metrics.get("train_f1"),
                "train_acc": fr.metrics.get("train_acc"),
                "n_samples": fr.num_examples,
            }
        self.history["train"][server_round] = per_tre
        return aggregated, {}

    # ── Aggregate evaluate ───────────────────────────────────────────────────
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_n = sum(r.num_examples for _, r in results)

        # Sample-weighted average of every test_* metric the clients returned
        metric_keys = ["test_auc", "test_acc", "test_f1",
                    "test_prec", "test_rec", "test_ll"]
        combined = {
            k: sum(r.num_examples * r.metrics.get(k, 0.0) for _, r in results) / total_n
            for k in metric_keys
        }
        global_ll = combined["test_ll"]

        per_tre = {}
        for _, er in results:
            tre_num = er.metrics.get("tre_num", 0)
            per_tre[tre_num] = {
                "cohort"   : er.metrics.get("cohort"),
                "test_auc" : er.metrics.get("test_auc"),
                "test_acc" : er.metrics.get("test_acc"),
                "test_f1"  : er.metrics.get("test_f1"),
                "test_prec": er.metrics.get("test_prec"),
                "test_rec" : er.metrics.get("test_rec"),
                "test_ll"  : er.metrics.get("test_ll"),
                "n_samples": er.num_examples,
            }

        self.history["eval"][server_round]      = per_tre
        self.history["global_ll"][server_round] = global_ll
        self.history["global"][server_round]    = combined   # new
        self._last_combined = combined                        # new

        server_eval_summary(server_round, combined, per_tre)
        server_round_footer()

        return global_ll, combined

    # ── Persistence at end of run ────────────────────────────────────────────
    def save_final_artifacts(self, n_features: int):
        """Write final model weights + history to disk."""
        os.makedirs(self.results_dir, exist_ok=True)

        if self._last_params is None:
            print("  [WARN] No final parameters to save (no rounds completed?)")
            return

        coef, intercept = self._last_params

        # Numpy weights (framework-agnostic)
        npz_path = os.path.join(self.results_dir, "prs_global_weights.npz")
        np.savez(npz_path, coef=coef, intercept=intercept)

        # Pickle (for direct sklearn use elsewhere)
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss="log_loss", penalty="l2",
                              class_weight="balanced", random_state=42)
        model.coef_      = coef.reshape(1, -1).copy()
        model.intercept_ = intercept.copy()
        model.classes_   = np.array([0, 1])
        pkl_path = os.path.join(self.results_dir, "prs_global_model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

        # JSON history + metadata
        meta = {
            "saved_at"       : datetime.now(timezone.utc).isoformat(),
            "model_type"     : "SGDClassifier (logistic regression, balanced)",
            "n_features"     : int(n_features),
            "n_rounds"       : self.n_rounds,
            "deployment_mode": "production (Flower client/server)",
            "history"        : self.history,
        }
        json_path = os.path.join(self.results_dir, "training_history.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        # if self._last_combined:
        #     c = self._last_combined
        #     print(f"\n  ── FINAL GLOBAL MODEL · federated test metrics ──")
        #     print(f"    AUC      : {c['test_auc']:.4f}")
        #     print(f"    F1       : {c['test_f1']:.4f}")
        #     print(f"    Accuracy : {c['test_acc']:.4f}")
        #     print(f"    Precision: {c['test_prec']:.4f}")
        #     print(f"    Recall   : {c['test_rec']:.4f}")
        #     print(f"    Log-loss : {c['test_ll']:.4f}")
        print(f"\n  Final model       → {pkl_path}")
        print(f"  Final weights     → {npz_path}")
        print(f"  Training history  → {json_path}")
