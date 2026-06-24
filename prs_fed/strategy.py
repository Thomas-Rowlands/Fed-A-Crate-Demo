"""
strategy.py — Custom FedAvg that captures per-TRE training/evaluation history.

The built-in FedAvg gives us:
  • Sample-weighted averaging of ArrayRecords (the model weights)
  • Sample-weighted averaging of MetricRecords (returned aggregated metrics)
  • All client sampling logic

What we add by subclassing:
  • Per-TRE-per-round history tracking (the default FedAvg only keeps the
    AGGREGATED MetricRecord, not the per-client breakdown we want to write
    to training_history.json).

If you don't need per-TRE history, drop this file entirely and instantiate
`flwr.serverapp.strategy.FedAvg(...)` directly in server_app.py.
"""

from flwr.app              import Message, MetricRecord
from flwr.serverapp.strategy import FedAvg


class HistoryFedAvg(FedAvg):
    """FedAvg that also tracks per-TRE metrics for each round."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Populated in-place each round so the ServerApp can persist it later.
        self.per_tre_history: dict[str, dict[int, dict]] = {
            "train": {},
            "evaluate": {},
        }

    # ── Capture per-TRE training metrics ─────────────────────────────────────
    # We override aggregate_train to peek at the replies before/after the
    # built-in FedAvg aggregation runs.
    def aggregate_train(self, server_round, replies):
        per_tre = {}
        for r in replies:
            if r.has_error():
                continue
            mr = r.content.get("metrics")
            if mr is None:
                continue
            tre_num = int(mr.get("tre-num", 0))
            per_tre[tre_num] = {
                "train-auc"   : float(mr.get("train-auc",  float("nan"))),
                "train-f1"    : float(mr.get("train-f1",   float("nan"))),
                "train-acc"   : float(mr.get("train-acc",  float("nan"))),
                "num-examples": int(mr.get("num-examples", 0)),
            }
        self.per_tre_history["train"][server_round] = per_tre

        # Delegate the actual aggregation (weight averaging + metric averaging)
        # to the parent FedAvg implementation.
        return super().aggregate_train(server_round, replies)

    # ── Capture per-TRE evaluation metrics ────────────────────────────────────
    def aggregate_evaluate(self, server_round, replies):
        per_tre = {}
        for r in replies:
            if r.has_error():
                continue
            mr = r.content.get("metrics")
            if mr is None:
                continue
            tre_num = int(mr.get("tre-num", 0))
            per_tre[tre_num] = {
                "test-auc"    : float(mr.get("test-auc",  float("nan"))),
                "test-acc"    : float(mr.get("test-acc",  float("nan"))),
                "test-f1"     : float(mr.get("test-f1",   float("nan"))),
                "test-prec"   : float(mr.get("test-prec", float("nan"))),
                "test-rec"    : float(mr.get("test-rec",  float("nan"))),
                "test-ll"     : float(mr.get("test-ll",   float("nan"))),
                "num-examples": int(mr.get("num-examples", 0)),
            }
        self.per_tre_history["evaluate"][server_round] = per_tre

        return super().aggregate_evaluate(server_round, replies)
