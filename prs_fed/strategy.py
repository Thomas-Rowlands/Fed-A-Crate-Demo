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
        # These are internal bookkeeping, NOT hyperparameters. They're stored
        # with a leading underscore so flwrCrate's strategy-attribute capture
        # (which records public attributes as hyperparameters) skips them —
        # otherwise they'd appear in the run-crate as empty/meaningless
        # "#strategy-param-*" entries captured at construction time.
        # Accessed via public properties below for readability in server_app.
        self._per_tre_history: dict[str, dict[int, dict]] = {
            "train": {},
            "evaluate": {},
        }
        # tre_num -> {"cohort": str, "crate_json": str, "present": bool}
        # The crate is identical every round; we keep the latest seen per TRE.
        self._provenance: dict[int, dict] = {}

    # NOTE: deliberately NO public @property accessors for these two.
    # flwrCrate's strategy-attribute capture iterates dir(strategy) and skips
    # only names starting with "_". A public property (e.g. `provenance`)
    # would show up in dir() and be captured as a bogus hyperparameter, which
    # is exactly what we're avoiding. Read them as `strategy._provenance` /
    # `strategy._per_tre_history` from server_app instead.

    def _capture_provenance(self, meta, tre_num: int, cohort: str) -> None:
        """Stash a TRE's RO-Crate string from its reply's meta ConfigRecord."""
        if meta is None:
            return
        crate_json = str(meta.get("ro-crate", ""))
        present    = bool(meta.get("ro-crate-present", False))
        self._provenance[tre_num] = {
            "cohort"    : cohort,
            "crate_json": crate_json,
            "present"   : present,
        }

    # ── Capture per-TRE training metrics ─────────────────────────────────────
    # We override aggregate_train to peek at the replies before/after the
    # built-in FedAvg aggregation runs.
    def aggregate_train(self, server_round, replies):
        per_tre = {}
        for r in replies:
            if r.has_error():
                continue
            mr   = r.content.get("metrics")
            meta = r.content.get("meta")        # ConfigRecord with identity
            if mr is None:
                continue
            # tre-num / cohort live in the non-aggregated ConfigRecord, not
            # in the MetricRecord (which FedAvg sample-weight-averages).
            tre_num = int(meta["tre-num"]) if meta is not None else 0
            cohort  = str(meta["cohort"])  if meta is not None else "?"
            self._capture_provenance(meta, tre_num, cohort)
            per_tre[tre_num] = {
                "cohort"      : cohort,
                "train-auc"   : float(mr.get("train-auc",  float("nan"))),
                "train-f1"    : float(mr.get("train-f1",   float("nan"))),
                "train-acc"   : float(mr.get("train-acc",  float("nan"))),
                "num-examples": int(mr.get("num-examples", 0)),
            }
        self._per_tre_history["train"][server_round] = per_tre

        # Delegate the actual aggregation (weight averaging + metric averaging)
        # to the parent FedAvg implementation.
        return super().aggregate_train(server_round, replies)

    # ── Capture per-TRE evaluation metrics ────────────────────────────────────
    def aggregate_evaluate(self, server_round, replies):
        per_tre = {}
        for r in replies:
            if r.has_error():
                continue
            mr   = r.content.get("metrics")
            meta = r.content.get("meta")
            if mr is None:
                continue
            tre_num = int(meta["tre-num"]) if meta is not None else 0
            cohort  = str(meta["cohort"])  if meta is not None else "?"
            self._capture_provenance(meta, tre_num, cohort)
            per_tre[tre_num] = {
                "cohort"      : cohort,
                "test-auc"    : float(mr.get("test-auc",  float("nan"))),
                "test-acc"    : float(mr.get("test-acc",  float("nan"))),
                "test-f1"     : float(mr.get("test-f1",   float("nan"))),
                "test-prec"   : float(mr.get("test-prec", float("nan"))),
                "test-rec"    : float(mr.get("test-rec",  float("nan"))),
                "test-ll"     : float(mr.get("test-ll",   float("nan"))),
                "num-examples": int(mr.get("num-examples", 0)),
            }
        self._per_tre_history["evaluate"][server_round] = per_tre

        return super().aggregate_evaluate(server_round, replies)