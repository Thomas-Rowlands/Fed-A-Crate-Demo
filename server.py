"""
server.py
---------
Custom FedAvg strategy that:
  • Aggregates linear-regression parameters using weighted averaging
    (weights = number of local training samples per client).
  • Logs per-round metrics (global MSE, per-cohort MSE / R²) to a shared
    results dict that the main script can read back for plotting.
"""

import numpy as np
from typing import Optional, Union
from functools import reduce

import flwr as fl
from flwr.common import (
    Parameters,
    FitIns, FitRes,
    EvaluateIns, EvaluateRes,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class PRSFedAvg(FedAvg):
    """
    FedAvg strategy extended to collect per-round metrics for post-hoc
    analysis and plotting.

    Extra constructor arguments
    ---------------------------
    n_features   : int  – used to initialise global model weights to zero
    results_log  : dict – populated in-place with round-by-round results
    cohort_labels: list – human-readable labels for each client
    """

    def __init__(
        self,
        n_features   : int,
        results_log  : dict,
        cohort_labels: list[str],
        **kwargs,
    ):
        # Zero-initialise the global model so round 1 clients start from a
        # common baseline rather than random local weights.
        initial_params = ndarrays_to_parameters(
            [np.zeros(n_features, dtype=np.float32), np.zeros(1, dtype=np.float32)]
        )
        super().__init__(initial_parameters=initial_params, **kwargs)

        self.n_features    = n_features
        self.results_log   = results_log
        self.cohort_labels = cohort_labels

    # ── Aggregation ─────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round : int,
        results      : list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict]:
        """Weighted-average of client model parameters."""
        if not results:
            return None, {}

        # Collect (num_examples, [coef, intercept]) for each client
        weights_results = [
            (fit_res.num_examples, parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        ]

        # Weighted average
        total = sum(n for n, _ in weights_results)
        agg_coef      = sum(n * p[0] for n, p in weights_results) / total
        agg_intercept = sum(n * p[1] for n, p in weights_results) / total

        aggregated = ndarrays_to_parameters([agg_coef, agg_intercept])

        # Log per-client training metrics for this round
        round_train = {}
        for i, (_, fit_res) in enumerate(results):
            label = fit_res.metrics.get("cohort", f"client_{i}")
            round_train[label] = {
                "train_mse": fit_res.metrics.get("train_mse"),
                "train_r2" : fit_res.metrics.get("train_r2"),
                "n_samples": fit_res.num_examples,
            }

        if "train" not in self.results_log:
            self.results_log["train"] = {}
        self.results_log["train"][server_round] = round_train

        return aggregated, {}

    # ── Evaluation ──────────────────────────────────────────────────────────

    def aggregate_evaluate(
        self,
        server_round : int,
        results      : list[tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> tuple[Optional[float], dict]:
        """Log per-client test metrics and compute weighted global MSE."""
        if not results:
            return None, {}

        total_n   = sum(eval_res.num_examples for _, eval_res in results)
        global_mse = sum(
            eval_res.num_examples * eval_res.metrics.get("test_mse", 0.0)
            for _, eval_res in results
        ) / total_n

        round_eval: dict[str, dict] = {}
        for i, (_, eval_res) in enumerate(results):
            label = eval_res.metrics.get("cohort", f"client_{i}")
            round_eval[label] = {
                "test_mse"  : eval_res.metrics.get("test_mse"),
                "test_r2"   : eval_res.metrics.get("test_r2"),
                "n_samples" : eval_res.num_examples,
            }

        if "eval" not in self.results_log:
            self.results_log["eval"] = {}
        self.results_log["eval"][server_round] = round_eval

        if "global_mse" not in self.results_log:
            self.results_log["global_mse"] = {}
        self.results_log["global_mse"][server_round] = global_mse

        print(
            f"\n  [Round {server_round:2d}]  Global weighted MSE = {global_mse:.4f}"
        )
        for label, m in round_eval.items():
            print(
                f"             {label:<10s}  MSE={m['test_mse']:.4f}  R²={m['test_r2']:.4f}"
            )

        return global_mse, {"global_mse": global_mse}
