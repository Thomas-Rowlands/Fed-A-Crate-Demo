"""
server_app.py — Server-side application.

Under the Message API, the ServerApp is a single function decorated with
@app.main() that orchestrates the whole federation. It is invoked by the
Flower runtime (SuperLink) when `flwr run` is called and runs to completion.

The function receives two things:
  • `grid`    — the communication channel to the connected SuperNodes
  • `context` — runtime config (the [tool.flwr.app.config] block from
                pyproject.toml, plus any --run-config overrides)

The job here is:
  1. Build an initial (zeroed) ArrayRecord of the right shape.
  2. Instantiate the strategy (HistoryFedAvg).
  3. Call strategy.start(...) — this is the blocking loop that does all
     federated training rounds and returns a `Result`.
  4. After it returns, persist the final model and history to disk.
"""

import os
from pathlib import Path

import numpy as np

from flwr.app                import ArrayRecord, ConfigRecord, Context
from flwr.serverapp          import Grid, ServerApp

from prs_fed.strategy        import HistoryFedAvg
from prs_fed.model_io        import save_final_artifacts
from prs_fed.crate_merge     import merge_tre_crates_into_run_crate

# flwrCrate is an optional dependency: if it's installed, the run emits an
# RO-Crate describing the whole federation. If it's not, the federation still
# runs and we just skip crate emission. This keeps the app runnable for anyone
# who doesn't have the (in-development) module.
try:
    from flwrcrate import FLCrateTracker
    _HAVE_FLWRCRATE = True
except ImportError:
    _HAVE_FLWRCRATE = False


app = ServerApp()


def _resolve_pyproject_path() -> str | None:
    """
    Locate pyproject.toml as an ABSOLUTE path.

    flwrCrate reads pyproject.toml to capture framework versions and the
    [tool.flwrcrate.metric-uris] mapping. The ServerApp runs from the
    installed app bundle (e.g. /app/.flwr/apps/<hash>/), NOT your project
    dir, so a relative "pyproject.toml" resolves against the wrong place.

    We anchor to the installed prs_fed package and look one level up, then
    fall back to a couple of other likely locations. Returns None if we
    genuinely can't find it (flwrCrate will then fall back to its own
    default and just log a warning about missing dependency capture).
    """
    candidates = []

    # 1. An explicit override via env var takes priority, if set.
    env_path = os.environ.get("PYPROJECT_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # 2. Beside the installed package: …/<hash>/pyproject.toml
    import prs_fed
    pkg_parent = Path(prs_fed.__file__).resolve().parent.parent
    candidates.append(pkg_parent / "pyproject.toml")

    # 3. Current working directory (covers `flwr run` from project root
    #    and local simulation).
    candidates.append(Path.cwd() / "pyproject.toml")

    # 4. The image's WORKDIR — compose.yml COPYs pyproject.toml to /app.
    #    This is the reliable location in the Docker deployment, since the
    #    ServerApp bundle dir may not include pyproject.toml.
    candidates.append(Path("/app/pyproject.toml"))

    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


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
    # If flwrCrate is available, wrap the run so it emits an RO-Crate of the
    # whole federation. Our app does client-side evaluation (no evaluate_fn
    # passed to start()), so per the flwrCrate docs we skip wrap_evaluate and
    # rely on record_result() reading the Result's client-side aggregates.
    crate_out = os.path.join(results_dir, "fl_crate")

    if _HAVE_FLWRCRATE:
        pyproject_path = _resolve_pyproject_path()
        if pyproject_path is None:
            print("  [flwrcrate] WARNING: could not locate pyproject.toml; "
                  "framework/dependency capture will be limited.")

        with FLCrateTracker(
            context, strategy,
            output_dir     = crate_out,                       # absolute (under /results)
            pyproject_path = pyproject_path or "pyproject.toml",
            app_name       = "Federated PRS case/control classification across 3 TREs",
            author         = _build_author(cfg),
            license        = "https://spdx.org/licenses/MIT.html",
        ) as tracker:
            result = strategy.start(
                grid            = grid,
                initial_arrays  = initial_arrays,
                num_rounds      = num_rounds,
                train_config    = train_config,
                evaluate_config = evaluate_config,
            )
            # No evaluate_fn was passed, so record_result reads per-round
            # metrics from the Result's client-side aggregates.
            tracker.record_result(result)
        print(f"  [flwrcrate] RO-Crate written to {crate_out}/ro-crate/")

        # ── Merge per-TRE provenance into the run-crate ──────────────────────
        # flwrCrate has now written its run-crate. Fold each TRE's RO-Crate
        # (institute + geo) into it as contributor Organizations on the run
        # action, producing a single self-contained provenance record.
        run_crate_path = os.path.join(crate_out, "ro-crate", "ro-crate-metadata.json")
        merge_status = merge_tre_crates_into_run_crate(
            run_crate_path = run_crate_path,
            tre_provenance = strategy._provenance,
        )
        if merge_status["written"]:
            merged = [t["cohort"] for t in merge_status["merged_tres"]]
            print(f"  [provenance] Merged TRE crates into run-crate: {merged}")
        if merge_status["skipped_tres"]:
            skipped = [(t["cohort"], t["reason"]) for t in merge_status["skipped_tres"]]
            print(f"  [provenance] Skipped: {skipped}")
        for w in merge_status["warnings"]:
            print(f"  [provenance] WARNING: {w}")
    else:
        print("  [flwrcrate] module not installed; skipping RO-Crate emission.")
        print("  [provenance] Without flwrCrate, no provenance crate is written. "
              "Install prs_fed/flwrcrate to enable provenance capture.")
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
        "per_tre"           : strategy._per_tre_history,        # our addition
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

def _build_author(cfg) -> dict | str | None:
    """
    Build the crate author from run-config, so it's not hard-coded.

    Reads author-name / author-orcid / author-affiliation from the
    [tool.flwr.app.config] block (override at run time with
    `flwr run . local-deployment --run-config "author-name='Jane Doe'"`).

    Returns a dict if a name is set, else None (flwrCrate then warns, which
    is the correct nudge to fill it in for a public release).
    """
    name = str(cfg.get("author-name", "")).strip()
    if not name:
        return None
    author = {"name": name}
    orcid = str(cfg.get("author-orcid", "")).strip()
    if orcid:
        author["orcid"] = orcid
    affil = str(cfg.get("author-affiliation", "")).strip()
    if affil:
        author["affiliation"] = affil
    return author


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