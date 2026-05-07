"""
server.py — Custom FedAvg strategy with TRE-flavoured logging.

All narrative output happens in the SERVER (main process) — the clients
themselves stay quiet because Ray prefixes their stdout with worker PIDs.
"""

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from tre_logger import (
    TRE_INFO, round_header, round_footer, aggregating, round_eval,
    server_to_tre, training_at_tre, tre_to_server,
)


class TREStrategy(FedAvg):
    """FedAvg with narrative server-side logging."""

    def __init__(self, n_features, results_log, n_rounds,
                 train_sample_counts=None, **kwargs):
        initial_params = ndarrays_to_parameters([
            np.zeros(n_features, dtype=np.float64),
            np.zeros(1,          dtype=np.float64),
        ])
        super().__init__(initial_parameters=initial_params, **kwargs)
        self.n_features         = n_features
        self.results_log        = results_log
        self.n_rounds           = n_rounds
        self.train_sample_counts = train_sample_counts or {}  # {cohort: n_train}

    # ── Round header + per-TRE "sending model" before fit ───────────────────
    def configure_fit(self, server_round, parameters, client_manager):
        round_header(server_round, self.n_rounds)
        # Server-side narrative: announce that the global model is being
        # distributed to all TREs in deterministic order so it reads cleanly.
        for cohort_label in ["USA_young", "USA_old", "USA_normal"]:
            if cohort_label in TRE_INFO:
                tre_num, name, colour = TRE_INFO[cohort_label]
                server_to_tre(tre_num, name, colour)
                n_samples = self.train_sample_counts.get(cohort_label, 0)
                training_at_tre(tre_num, name, colour, n_samples, 5)
        return super().configure_fit(server_round, parameters, client_manager)

    # ── After local training: print per-TRE return + aggregate ──────────────
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Print one "TRE → Server" line per result, in TRE-number order
        ordered = sorted(
            results,
            key=lambda r: r[1].metrics.get("tre_num", 99),
        )
        for _, fr in ordered:
            cohort = fr.metrics.get("cohort", "?")
            if cohort in TRE_INFO:
                tre_num, _, colour = TRE_INFO[cohort]
                tre_to_server(
                    tre_num, colour,
                    auc=fr.metrics.get("train_auc"),
                    f1 =fr.metrics.get("train_f1"),
                )

        # FedAvg
        weights = [(r.num_examples, parameters_to_ndarrays(r.parameters))
                   for _, r in results]
        total      = sum(n for n, _ in weights)
        agg_coef   = sum(n * p[0] for n, p in weights) / total
        agg_int    = sum(n * p[1] for n, p in weights) / total
        aggregated = ndarrays_to_parameters([agg_coef, agg_int])

        self._last_parameters = [agg_coef, agg_int]
        aggregating(len(results))

        # Save train metrics for plotting
        round_train = {}
        for _, fr in results:
            label = fr.metrics.get("cohort", "?")
            round_train[label] = {
                "train_auc": fr.metrics.get("train_auc"),
                "train_f1" : fr.metrics.get("train_f1"),
                "train_acc": fr.metrics.get("train_acc"),
            }
        self.results_log.setdefault("train", {})[server_round] = round_train
        return aggregated, {}

    # ── Aggregate evaluation: print eval block, end-of-round footer ─────────
    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_n   = sum(r.num_examples for _, r in results)
        global_ll = sum(r.num_examples * r.metrics.get("test_ll", 0.0)
                        for _, r in results) / total_n

        round_eval_dict = {}
        for _, er in results:
            label = er.metrics.get("cohort", "?")
            round_eval_dict[label] = {
                "test_auc" : er.metrics.get("test_auc"),
                "test_acc" : er.metrics.get("test_acc"),
                "test_f1"  : er.metrics.get("test_f1"),
                "test_prec": er.metrics.get("test_prec"),
                "test_rec" : er.metrics.get("test_rec"),
                "test_ll"  : er.metrics.get("test_ll"),
            }

        # Sort the eval dict by TRE number so it always prints 1, 2, 3
        ordered_dict = {}
        for cohort in ["USA_young", "USA_old", "USA_normal"]:
            if cohort in round_eval_dict:
                ordered_dict[cohort] = round_eval_dict[cohort]

        self.results_log.setdefault("eval", {})[server_round]      = round_eval_dict
        self.results_log.setdefault("global_ll", {})[server_round] = global_ll

        round_eval(server_round, global_ll, ordered_dict)
        round_footer()

        return global_ll, {"global_ll": global_ll}
