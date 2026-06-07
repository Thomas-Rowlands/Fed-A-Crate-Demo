"""
main.py — Federated PRS demo with TRE-narrative console output.

Demonstrates training a polygenic risk score classifier across three Trusted
Research Environments (TREs) without data leaving each TRE.  Only model
weights cross the boundary; the central server aggregates them via FedAvg.

Run:
    python app/main.py --data-dir data
"""

# ── Suppress third-party log noise BEFORE importing flwr/ray ────────────────
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["RAY_DEDUP_LOGS"]               = "1"
os.environ["RAY_DISABLE_IMPORT_WARNING"]   = "1"
os.environ["RAY_LOG_TO_STDERR"]            = "0"
os.environ["PYTHONWARNINGS"]               = "ignore"
os.environ["RAY_DEDUP_LOGS_AGG_WINDOW_S"]  = "0"
# Disable Ray's metrics export (otherwise it logs 'failed to connect to metrics agent')
os.environ["RAY_enable_metrics_collection"] = "false"
os.environ["RAY_metrics_report_interval_ms"] = "0"
os.environ["GRPC_VERBOSITY"]               = "ERROR"
os.environ["GRPC_TRACE"]                   = ""

import argparse
import logging
import numpy as np

# Silence Flower's loggers before flwr is imported
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("ray").setLevel(logging.ERROR)

import flwr as fl
from scipy.stats   import pearsonr
from scipy.special import expit
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                              precision_score, recall_score, f1_score)
from simulation.app.data_utils import load_all_datasets, recover_true_betas
from simulation.app.client     import make_client_fn
from simulation.app.server     import TREStrategy
from simulation.app.plotting   import generate_report
from simulation.app.tre_logger import (
    silence_third_party_logs, banner, section,
    cohort_summary_table, final_metrics_table, coef_recovery_table, done,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--rounds",     type=int, default=10)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def _evaluate_combined(dfs_test, fed_coef, fed_int):
    """Compute final metrics on the union of all three TRE test sets."""
    X_all = np.vstack([d["X_test"] for d in dfs_test])
    y_all = np.concatenate([d["y_test"] for d in dfs_test])
    probs = expit(X_all @ fed_coef + fed_int[0])
    preds = (probs >= 0.5).astype(int)
    return dict(
        accuracy = float(accuracy_score(y_all, preds)),
        auc      = float(roc_auc_score(y_all, probs)),
        log_loss = float(log_loss(y_all, probs)),
        precision= float(precision_score(y_all, preds, average="macro", zero_division=0)),
        recall   = float(recall_score(y_all, preds, average="macro", zero_division=0)),
        f1       = float(f1_score(y_all, preds, average="macro", zero_division=0)),
    )


def main():
    args = parse_args()

    # Hide third-party log noise
    silence_third_party_logs()

    banner(
        "Federated PRS Case/Control Prediction across 3 TREs",
        f"Flower · FedAvg · Logistic Regression  ·  Rounds: {args.rounds}",
    )

    # ── Step 1: Load datasets at each TRE ────────────────────────────────────
    section(1, 4, "Loading datasets at each TRE")
    datasets      = load_all_datasets(data_dir=args.data_dir)
    cohort_labels = ["USA_young", "USA_old", "USA_normal"]
    metas         = [d[5] for d in datasets]
    n_features    = datasets[0][0].shape[1]
    cohort_summary_table(metas)

    # ── Step 2: Recover true PRS effect sizes (for Figure 3 comparison) ──────
    section(2, 4, "Recovering true PRS effect sizes (for evaluation only)")
    ref_csv    = os.path.join(args.data_dir, "USA_normal.csv")
    true_betas = recover_true_betas(ref_csv)
    print(f"  {len(true_betas)} SNP betas recovered from reference dataset.")
    print("  (These never participate in training; used only to score the federated model.)")

    # ── Step 3: Run federated training ───────────────────────────────────────
    section(3, 4, f"Running federated training across 3 TREs ({args.rounds} rounds)")
    print("  Each round: Server distributes the global model to each TRE,")
    print("  each TRE trains locally on its private data, then returns weights")
    print("  (not data) to the server, which aggregates them with FedAvg.")

    results_log = {}
    train_counts = {label: datasets[i][0].shape[0]
                    for i, label in enumerate(cohort_labels)}
    strategy    = TREStrategy(
        n_features           = n_features,
        results_log          = results_log,
        n_rounds             = args.rounds,
        train_sample_counts  = train_counts,
        fraction_fit         = 1.0,
        fraction_evaluate    = 1.0,
        min_fit_clients      = 3,
        min_evaluate_clients = 3,
        min_available_clients= 3,
    )
    client_fn = make_client_fn(datasets, cohort_labels)

    # Open devnull and redirect stderr at the FILE DESCRIPTOR level
    # (Ray's C++ workers bypass Python's sys.stderr entirely; only fd-level
    # redirection silences them.)
    devnull_fd     = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)              # back up original stderr fd
    os.dup2(devnull_fd, 2)                   # point fd 2 (stderr) to /dev/null

    try:
        fl.simulation.start_simulation(
            client_fn        = client_fn,
            num_clients      = 3,
            config           = fl.server.ServerConfig(num_rounds=args.rounds),
            strategy         = strategy,
            client_resources = {"num_cpus": 1, "num_gpus": 0.0},
        )

        final_params = strategy._last_parameters

        # ── Step 4: Final evaluation, save model, generate plots ─────────────
        section(4, 4, "Saving global model & generating plots")

        test_views = [
            {"X_test": d[1], "y_test": d[3]} for d in datasets
        ]
        metrics = _evaluate_combined(test_views, final_params[0], final_params[1])
        final_metrics_table(metrics)

        coef = final_params[0]
        r, p = pearsonr(true_betas, coef)
        rmse = float(np.sqrt(np.mean((true_betas - coef) ** 2)))
        mask = np.abs(true_betas) > 1e-8
        mape = float(np.mean(np.abs((true_betas[mask] - coef[mask]) / true_betas[mask])) * 100)
        coef_recovery_table(
            dict(pearson_r=r, rmse=rmse, mape=mape, n=len(coef)),
        )

        output_paths = generate_report(
            data_dir     = args.data_dir,
            output_dir   = args.output_dir,
            cohort_metas = metas,
            n_rounds     = args.rounds,
        )

        done(output_paths)
    finally:
        # Keep stderr suppressed (Ray shutdown still emits errors after
        # we exit this function). The OS will clean up on process exit.
        os.close(devnull_fd)
        os.close(saved_stderr_fd)


if __name__ == "__main__":
    main()
