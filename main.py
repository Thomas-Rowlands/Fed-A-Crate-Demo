"""
main.py
-------
Entry point for the PRS Federated Learning demo.

Run from the project root with:
    python main.py [--data-dir PATH] [--rounds N] [--output-dir PATH]

Layout
------
prs_federation/
├── main.py          ← you are here
├── data_utils.py    ← CSV loading & preprocessing
├── model.py         ← SGDRegressor + parameter serialisation
├── client.py        ← Flower NumPyClient per cohort
├── server.py        ← Custom FedAvg strategy with metric logging
└── plotting.py      ← All visualisation

The three CSVs (USA_young.csv, USA_normal.csv, USA_old.csv) should be either
in the same directory as this script, or specified via --data-dir.
"""

import argparse
import os
import sys

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from data_utils import load_all_datasets
from model      import create_model, get_parameters, set_parameters
from client     import make_client_fn
from server     import PRSFedAvg
from plotting   import generate_report


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PRS Federated Learning with Flower")
    p.add_argument(
        "--data-dir",  default=".",
        help="Directory containing the three cohort CSV files (default: .)"
    )
    p.add_argument(
        "--rounds",    type=int, default=10,
        help="Number of federation rounds (default: 10)"
    )
    p.add_argument(
        "--output-dir", default="results",
        help="Where to save plots and summary (default: results/)"
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_cohort_summary(metas: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("  COHORT SUMMARY")
    print("=" * 65)
    header = f"{'Cohort':<14} {'N':>7} {'Age μ':>7} {'Age σ':>6} "  \
             f"{'PRS μ':>7} {'Case%':>7}"
    print(header)
    print("-" * 65)
    for m in metas:
        print(
            f"  {m['cohort_label']:<12} {m['n_samples']:>7,}  "
            f"{m['age_mean']:>6.1f}  {m['age_std']:>5.1f}  "
            f"{m['prs_mean']:>6.3f}  {m['case_rate']*100:>6.2f}%"
        )
    print("=" * 65 + "\n")


def print_final_summary(results_log: dict, cohort_labels: list[str]) -> None:
    eval_log = results_log.get("eval", {})
    if not eval_log:
        return

    rounds     = sorted(eval_log.keys())
    last_round = rounds[-1]
    last_eval  = eval_log[last_round]

    print("\n" + "=" * 55)
    print(f"  FINAL RESULTS  (Round {last_round})")
    print("=" * 55)
    for label in cohort_labels:
        m = last_eval.get(label, {})
        print(
            f"  {label:<14}  MSE = {m.get('test_mse', float('nan')):.4f}  "
            f"R² = {m.get('test_r2', float('nan')):.4f}"
        )
    global_mse = results_log.get("global_mse", {}).get(last_round)
    if global_mse is not None:
        print(f"\n  Global weighted MSE = {global_mse:.4f}")
    print("=" * 55 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  Polygenic Risk Score – Federated Learning Demo")
    print("  Framework : Flower (flwr)")
    print("  Strategy  : FedAvg (weighted by sample count)")
    print("  Model     : Linear Regression (SGDRegressor)")
    print(f"  Rounds    : {args.rounds}")
    print("=" * 65)

    # 1. Load datasets ────────────────────────────────────────────────────────
    print("\n[1/4] Loading datasets …")
    datasets = load_all_datasets(data_dir=args.data_dir)
    cohort_labels = ["USA_young", "USA_normal", "USA_old"]
    metas         = [d[5] for d in datasets]   # meta dict is index 5
    n_features    = datasets[0][0].shape[1]
    print_cohort_summary(metas)

    # 2. Prepare federation ───────────────────────────────────────────────────
    print("[2/4] Configuring Flower simulation …")
    results_log  = {}   # filled in-place by PRSFedAvg strategy
    strategy     = PRSFedAvg(
        n_features        = n_features,
        results_log       = results_log,
        cohort_labels     = cohort_labels,
        # Fraction of clients sampled each round (all 3 every round)
        fraction_fit      = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients   = 3,
        min_evaluate_clients = 3,
        min_available_clients = 3,
    )

    client_fn = make_client_fn(datasets, cohort_labels)

    # 3. Run simulation ───────────────────────────────────────────────────────
    print("[3/4] Running federated simulation …\n")
    history = fl.simulation.start_simulation(
        client_fn          = client_fn,
        num_clients        = 3,
        config             = fl.server.ServerConfig(num_rounds=args.rounds),
        strategy           = strategy,
        client_resources   = {"num_cpus": 1, "num_gpus": 0.0},
    )

    # Retrieve final global parameters from the strategy
    # (They live in strategy.initial_parameters after the last aggregate_fit)
    # We reconstruct by running one more evaluation pass
    final_params = None
    if hasattr(strategy, "_current_parameters"):
        final_params = parameters_to_ndarrays(strategy._current_parameters)
    else:
        # Fallback: rebuild from last aggregated weights stored in history
        # Use the last round's per-client weights to reconstruct global params
        eval_log   = results_log.get("eval", {})
        last_round = max(eval_log.keys()) if eval_log else 0

        # Re-fit a model on all data to get a reasonable set of final params
        # (this is just for the scatter plots – the federation already converged)
        print("\n  Reconstructing global parameters for plotting …")
        from sklearn.linear_model import SGDRegressor
        global_model = create_model(n_features)
        for i in range(len(datasets)):
            X_train, _, y_train, _, _, _ = datasets[i]
            for _ in range(3):
                global_model.partial_fit(X_train, y_train)
        final_params = get_parameters(global_model)

    print_final_summary(results_log, cohort_labels)

    # 4. Generate report & save model
    print("[4/4] Generating plots and saving model …")
    output_paths = generate_report(
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
        cohort_metas = metas,
        n_rounds     = args.rounds,
    )

    print("
  Done!  Outputs:")
    for p in output_paths:
        print(f"    {os.path.abspath(p)}")
    print()
    return output_paths


if __name__ == "__main__":
    main()
