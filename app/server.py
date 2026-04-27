"""
server.py
---------
Custom FedAvg strategy for federated logistic regression.
Logs per-round classification metrics (AUC, F1, Accuracy, Log Loss)
and aggregates using weighted averaging by n_samples per client.
"""

import numpy as np
from typing import Optional
import flwr as fl
from flwr.common import (
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg


class PRSFedAvg(FedAvg):

    def __init__(self, n_features, results_log, cohort_labels, **kwargs):
        initial_params = ndarrays_to_parameters([
            np.zeros(n_features, dtype=np.float64),
            np.zeros(1,          dtype=np.float64),
        ])
        super().__init__(initial_parameters=initial_params, **kwargs)
        self.n_features   = n_features
        self.results_log  = results_log
        self.cohort_labels= cohort_labels

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights = [(r.num_examples, parameters_to_ndarrays(r.parameters))
                   for _, r in results]
        total       = sum(n for n, _ in weights)
        agg_coef    = sum(n * p[0] for n, p in weights) / total
        agg_int     = sum(n * p[1] for n, p in weights) / total
        aggregated  = ndarrays_to_parameters([agg_coef, agg_int])

        # Store the latest global parameters for model saving
        self._last_parameters = [agg_coef, agg_int]

        round_train = {}
        for _, fit_res in results:
            label = fit_res.metrics.get("cohort", "?")
            round_train[label] = {
                "train_auc": fit_res.metrics.get("train_auc"),
                "train_f1" : fit_res.metrics.get("train_f1"),
                "train_acc": fit_res.metrics.get("train_acc"),
                "n_samples": fit_res.num_examples,
            }
        self.results_log.setdefault("train", {})[server_round] = round_train
        return aggregated, {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_n  = sum(r.num_examples for _, r in results)
        # Weighted global log-loss
        global_ll = sum(r.num_examples * r.metrics.get("test_ll", 0.0)
                        for _, r in results) / total_n

        round_eval = {}
        for _, eval_res in results:
            label = eval_res.metrics.get("cohort", "?")
            round_eval[label] = {
                "test_auc" : eval_res.metrics.get("test_auc"),
                "test_acc" : eval_res.metrics.get("test_acc"),
                "test_f1"  : eval_res.metrics.get("test_f1"),
                "test_prec": eval_res.metrics.get("test_prec"),
                "test_rec" : eval_res.metrics.get("test_rec"),
                "test_ll"  : eval_res.metrics.get("test_ll"),
                "n_samples": eval_res.num_examples,
            }

        self.results_log.setdefault("eval", {})[server_round] = round_eval
        self.results_log.setdefault("global_ll", {})[server_round] = global_ll

        print(f"\n  [Round {server_round:2d}]  Global Log-Loss = {global_ll:.4f}")
        for label, m in round_eval.items():
            print(f"    {label:<12}  AUC={m['test_auc']:.4f}  "
                  f"F1={m['test_f1']:.4f}  Acc={m['test_acc']:.4f}")

        return global_ll, {"global_ll": global_ll}
