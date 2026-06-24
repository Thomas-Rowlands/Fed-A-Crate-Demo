"""
server_app.py — Server-side application.

Under the Message API, the ServerApp is a single function decorated with
@app.main() that orchestrates the whole federation. It is invoked by the
Flower runtime (SuperLink) when `flwr run` is called and runs to completion.
"""

import numpy as np

from flwr.app                import ArrayRecord, ConfigRecord, Context
from flwr.serverapp          import Grid, ServerApp

from prs_fed.strategy        import HistoryFedAvg
from prs_fed.model_io        import save_final_artifacts


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # ── Pull config (with sensible defaults) ──────────────────────────────────
    cfg = context.run_config
    num_rounds   = int(cfg.get("num-server-rounds", 10))
    local_epochs = int(cfg.get("local-epochs",       5))
    n_features   = int(cfg.get("n-features",       313))
    threshold    = float(cfg.get("threshold",       0.5))
    results_dir  = str(cfg.get("results-dir", "/results"))

    # ── Initial global model parameters (all zeros) ───────────────────────────
    # The ClientApp's @app.train will use these as the starting point.
    # An ArrayRecord wraps an ordered list of numpy arrays under a single key.
    initial_arrays = ArrayRecord([
        np.zeros(n_features, dtype=np.float64),   # coef
        np.zeros(1,          dtype=np.float64),   # intercept
    ])

    # ── Per-round config sent to clients ─────────────────────────────────────
    # ClientApp reads these from msg.content["config"].
    train_config    = ConfigRecord({"local-epochs": local_epochs})
    evaluate_config = ConfigRecord({"threshold":    threshold})

    # ── Strategy ──────────────────────────────────────────────────────────────
    # The renamed args matter here: it's now `min_train_nodes` not
    # `min_fit_clients`, and `fraction_train` not `fraction_fit`.
    strategy = HistoryFedAvg(
        fraction_train      = 1.0,
        fraction_evaluate   = 1.0,
        min_train_nodes     = 3,
        min_evaluate_nodes  = 3,
        min_available_nodes = 3,
    )
    strategy.summary()

    # ── Run the federation ────────────────────────────────────────────────────
    result = strategy.start(
        grid            = grid,
        initial_arrays  = initial_arrays,
        num_rounds      = num_rounds,
        train_config    = train_config,
        evaluate_config = evaluate_config,
    )

    # ── Persist artefacts ─────────────────────────────────────────────────────
    # result.arrays is the final aggregated ArrayRecord from the last round.
    coef, intercept = result.arrays.to_numpy_ndarrays()

    # Merge Flower's aggregated history with our per-TRE history.
    full_history = {
        "per_tre"           : strategy.per_tre_history,         # our addition
        "aggregated_train"  : _round_records_to_dict(result.train_metrics_clientapp),
        "aggregated_evaluate": _round_records_to_dict(result.evaluate_metrics_clientapp),
    }

    paths = save_final_artifacts(
        coef        = coef,
        intercept   = intercept,
        history     = full_history,
        n_features  = n_features,
        n_rounds    = num_rounds,
        results_dir = results_dir,
    )

    print("\nFinal artefacts written:")
    for label, path in paths.items():
        print(f"  {label:>8}: {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _round_records_to_dict(round_records) -> dict:
    """
    Flower's Result returns metric history as a dict {round_num: MetricRecord}.
    Convert each MetricRecord to a plain dict for JSON serialisation.
    """
    if round_records is None:
        return {}
    return {
        int(r): {k: (v if isinstance(v, (int, float, str, bool)) else list(v))
                 for k, v in mr.items()}
        for r, mr in round_records.items()
    }
