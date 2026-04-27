"""
main.py — Federated PRS Case/Control Classification Demo
=========================================================
Scenario: Age-imbalanced federated clients, logistic regression,
coefficient recovery analysis vs true PRS effect sizes.

Run:
    python main.py [--data-dir PATH] [--rounds N] [--output-dir PATH]
"""

import argparse, os, sys
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays

from data_utils  import load_all_datasets, recover_true_betas
from model       import create_model, get_parameters, set_parameters, evaluate_model
from client      import make_client_fn
from server      import PRSFedAvg
from plotting    import generate_report
from scipy.stats import pearsonr
from scipy.special import expit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default=".")
    p.add_argument("--rounds",     type=int, default=10)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def print_cohort_summary(metas):
    print("\n" + "="*70)
    print("  COHORT SUMMARY  (Scenario: strong age imbalance)")
    print("="*70)
    print(f"  {'Cohort':<14} {'N':>7} {'<41y':>7} {'>69y':>7} "
          f"{'Cases':>7} {'Controls':>9} {'Case%':>7}")
    print("-"*70)
    for m in metas:
        print(f"  {m['cohort_label']:<14} {m['n_samples']:>7,} "
              f"{m['pct_under41']:>6.1f}% {m['pct_over69']:>6.1f}% "
              f"{m['n_cases']:>7} {m['n_controls']:>9,} "
              f"{m['case_rate']*100:>6.2f}%")
    print("="*70 + "\n")


def print_final_summary(results_log, cohort_labels, final_params, true_betas):
    eval_log = results_log.get("eval", {})
    if not eval_log:
        return
    last = eval_log[max(eval_log.keys())]

    print("\n" + "="*60)
    print("  FINAL CLASSIFICATION METRICS  (combined test set)")
    print("="*60)
    for label in cohort_labels:
        m = last.get(label, {})
        print(f"  {label:<14}  AUC={m.get('test_auc', float('nan')):.4f}  "
              f"F1={m.get('test_f1', float('nan')):.4f}  "
              f"Acc={m.get('test_acc', float('nan')):.4f}")

    if final_params and true_betas is not None:
        coef = final_params[0]
        r, p = pearsonr(true_betas, coef)
        rmse = float(np.sqrt(np.mean((true_betas - coef)**2)))
        mask = np.abs(true_betas) > 1e-8
        mape = float(np.mean(np.abs((true_betas[mask]-coef[mask])/true_betas[mask]))*100)
        print()
        print("  COEFFICIENT RECOVERY (vs true PRS effect sizes)")
        print(f"  Pearson r = {r:.4f}   p = {p:.2e}")
        print(f"  RMSE      = {rmse:.6f}")
        print(f"  MAPE      = {mape:.2f}%")
        print(f"  N variants= {len(coef)}")
        print()
        print("  Documented targets:")
        print("  Pearson r=0.8168  RMSE=0.051481  MAPE=76.15%")
        print("  AUC=0.6396  Acc=0.5970  F1=0.5492")
    print("="*60 + "\n")


def main():
    args = parse_args()

    print("\n" + "="*65)
    print("  Federated PRS Case/Control Prediction")
    print("  Framework : Flower (flwr)")
    print("  Strategy  : FedAvg (weighted by n_samples)")
    print("  Model     : Logistic Regression (SGDClassifier, balanced)")
    print("  Target    : case/control (binary)")
    print(f"  Rounds    : {args.rounds}")
    print("="*65)

    print("\n[1/4] Loading datasets …")
    datasets     = load_all_datasets(data_dir=args.data_dir)
    cohort_labels= ["USA_young", "USA_old", "USA_normal"]
    metas        = [d[5] for d in datasets]
    n_features   = datasets[0][0].shape[1]
    print_cohort_summary(metas)

    print("[2/4] Recovering true PRS effect sizes …")
    ref_csv    = os.path.join(args.data_dir, "USA_normal.csv")
    true_betas = recover_true_betas(ref_csv)
    print(f"  {len(true_betas)} SNP betas recovered from reference dataset.\n")

    print("[3/4] Configuring & running Flower simulation …\n")
    results_log = {}
    strategy    = PRSFedAvg(
        n_features           = n_features,
        results_log          = results_log,
        cohort_labels        = cohort_labels,
        fraction_fit         = 1.0,
        fraction_evaluate    = 1.0,
        min_fit_clients      = 3,
        min_evaluate_clients = 3,
        min_available_clients= 3,
    )
    client_fn = make_client_fn(datasets, cohort_labels)

    fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = 3,
        config           = fl.server.ServerConfig(num_rounds=args.rounds),
        strategy         = strategy,
        client_resources = {"num_cpus": 1, "num_gpus": 0.0},
    )

    # Retrieve final global parameters from strategy
    final_params = getattr(strategy, "_last_parameters", None)
    print_final_summary(results_log, cohort_labels, final_params, true_betas)

    print("[4/4] Generating plots and saving model …")
    output_paths = generate_report(
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
        cohort_metas = metas,
        n_rounds     = args.rounds,
    )

    print("\n  Done!  Outputs:")
    for p in output_paths:
        print(f"    {os.path.abspath(p)}")
    print()
    return output_paths


if __name__ == "__main__":
    main()
