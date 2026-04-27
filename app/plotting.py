"""
plotting.py
-----------
Visualisation for the federated PRS case/control classification demo.

Page 1 — Cohort characteristics & federation mechanics
Page 2 — Figure 3 equivalent: true betas vs federated coefficients,
          classification metrics, cross-cohort AUC heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score,
                              precision_score, recall_score, f1_score)
from scipy.special import expit
from scipy.stats import pearsonr

C = {
    "USA_young" : "#3B82F6",
    "USA_normal": "#10B981",
    "USA_old"   : "#F59E0B",
    "federated" : "#EF4444",
}
DISPLAY = {
    "USA_young" : "Dataset A – Young (μ=43 y)",
    "USA_normal": "Dataset C – Normal (30–80 y)",
    "USA_old"   : "Dataset B – Old (μ=61 y)",
    "federated" : "Federated (global)",
}
NAMES = ["USA_young", "USA_old", "USA_normal"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load(data_dir):
    dfs = {}
    for name in NAMES:
        df       = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
        snp_cols = [c for c in df.columns if ":" in c]
        X        = df[snp_cols].values.astype(np.float64)
        y        = df["case"].values.astype(int)
        sc       = StandardScaler()
        Xs       = sc.fit_transform(X)
        sp       = int(len(Xs) * 0.8)
        dfs[name] = dict(X_tr=Xs[:sp], X_te=Xs[sp:],
                         y_tr=y[:sp],  y_te=y[sp:],
                         scaler=sc, df=df, snp_cols=snp_cols)
    # True betas from normal dataset (reference)
    ref  = dfs["USA_normal"]
    betas = Ridge(alpha=1e-10).fit(
        ref["df"][ref["snp_cols"]].values.astype(np.float64),
        ref["df"]["prs"].values.astype(np.float64)
    ).coef_
    return dfs, betas


def _simulate_federation(dfs, n_rounds=10, local_epochs=5):
    np.random.seed(42)
    n_features   = dfs[NAMES[0]]["X_tr"].shape[1]
    global_coef  = np.zeros(n_features)
    global_int   = np.zeros(1)
    prev_coef    = global_coef.copy()

    round_metrics  = {n: {"auc":[], "f1":[], "acc":[], "ll":[]} for n in NAMES}
    global_ll_log  = []
    norm_changes   = []

    for _ in range(n_rounds):
        new_coefs, new_ints = [], []
        for name in NAMES:
            m = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                              learning_rate="invscaling", eta0=0.01,
                              max_iter=1, warm_start=True, random_state=42,
                              class_weight="balanced")
            m.coef_      = global_coef.reshape(1, -1).copy()
            m.intercept_ = global_int.copy()
            m.classes_   = np.array([0, 1])
            for _ in range(local_epochs):
                m.partial_fit(dfs[name]["X_tr"], dfs[name]["y_tr"], classes=[0,1])
            new_coefs.append(m.coef_[0].copy())
            new_ints.append(m.intercept_.copy())

        global_coef = np.mean(new_coefs, axis=0)
        global_int  = np.mean(new_ints,  axis=0)
        norm_changes.append(np.linalg.norm(global_coef - prev_coef))
        prev_coef   = global_coef.copy()

        total_ll = 0
        for name in NAMES:
            probs = expit(dfs[name]["X_te"] @ global_coef + global_int[0])
            preds = (probs >= 0.5).astype(int)
            y_te  = dfs[name]["y_te"]
            round_metrics[name]["auc"].append(float(roc_auc_score(y_te, probs)))
            round_metrics[name]["f1"].append(float(f1_score(y_te, preds, zero_division=0)))
            round_metrics[name]["acc"].append(float(accuracy_score(y_te, preds)))
            ll = float(log_loss(y_te, probs))
            round_metrics[name]["ll"].append(ll)
            total_ll += ll
        global_ll_log.append(total_ll / 3)

    return dict(global_coef=global_coef, global_int=global_int,
                round_metrics=round_metrics, global_ll_log=global_ll_log,
                norm_changes=norm_changes)


def _local_models(dfs):
    models = {}
    for name in NAMES:
        m = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          learning_rate="invscaling", eta0=0.01,
                          max_iter=1, warm_start=True, random_state=42,
                          class_weight="balanced")
        m.classes_ = np.array([0, 1])
        for _ in range(10):
            m.partial_fit(dfs[name]["X_tr"], dfs[name]["y_tr"], classes=[0,1])
        models[name] = m
    return models


# ── Plot functions ────────────────────────────────────────────────────────────

def _plot_age_dist(dfs, ax):
    for name in NAMES:
        ages = dfs[name]["df"]["ageOfEntry"].values
        ax.hist(ages, bins=40, alpha=0.5, color=C[name],
                label=DISPLAY[name], density=True, edgecolor="none")
        ax.axvline(ages.mean(), color=C[name], linewidth=2, linestyle="--")
    ax.set_title("Age-of-Entry Distribution\n(Intentional Cohort Imbalance)", fontweight="bold")
    ax.set_xlabel("Age of Entry"); ax.set_ylabel("Density")
    ax.legend(fontsize=7.5); sns.despine(ax=ax)


def _plot_case_rate(dfs, ax):
    labels = [DISPLAY[n] for n in NAMES]
    rates  = [dfs[n]["df"]["case"].mean() * 100 for n in NAMES]
    counts = [(dfs[n]["df"]["case"]==1).sum() for n in NAMES]
    bars   = ax.bar(labels, rates, color=[C[n] for n in NAMES], alpha=0.85, edgecolor="white")
    for bar, rate, cnt in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.2f}%\n(n={cnt})", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title("Case Rate by Cohort", fontweight="bold")
    ax.set_ylabel("Case Rate (%)"); ax.set_ylim(0, max(rates) * 1.5)
    ax.tick_params(axis="x", labelsize=7.5); sns.despine(ax=ax)


def _plot_norm_convergence(fed, ax):
    rounds = list(range(1, len(fed["norm_changes"]) + 1))
    ax.plot(rounds, fed["norm_changes"], "o-", color=C["federated"], linewidth=2.5, markersize=6)
    ax.fill_between(rounds, 0, fed["norm_changes"], alpha=0.15, color=C["federated"])
    ax.set_title("Parameter Convergence\n‖Δ global coef‖ per Round", fontweight="bold")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("L2 Norm of Δcoef")
    ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_auc_rounds(fed, ax):
    rounds = list(range(1, len(fed["global_ll_log"]) + 1))
    for name in NAMES:
        ax.plot(rounds, fed["round_metrics"][name]["auc"],
                "o-", color=C[name], linewidth=2, markersize=4, label=DISPLAY[name])
    ax.axhline(0.5, color="grey", linewidth=1, linestyle=":", label="Random baseline")
    ax.set_title("AUC per Federation Round\n(Global model on each cohort test set)", fontweight="bold")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=7); ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_f1_rounds(fed, ax):
    rounds = list(range(1, len(fed["global_ll_log"]) + 1))
    for name in NAMES:
        ax.plot(rounds, fed["round_metrics"][name]["f1"],
                "o-", color=C[name], linewidth=2, markersize=4, label=DISPLAY[name])
    ax.set_title("F1 Score per Federation Round\n(Global model on each cohort test set)", fontweight="bold")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("F1 Score")
    ax.legend(fontsize=7); ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_ll_rounds(fed, ax):
    rounds = list(range(1, len(fed["global_ll_log"]) + 1))
    for name in NAMES:
        ax.plot(rounds, fed["round_metrics"][name]["ll"],
                "o-", color=C[name], linewidth=2, markersize=4, label=DISPLAY[name])
    ax.plot(rounds, fed["global_ll_log"], "s--", color=C["federated"],
            linewidth=2.5, markersize=6, label="Global avg")
    ax.set_title("Log-Loss per Federation Round", fontweight="bold")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("Log Loss")
    ax.legend(fontsize=7); ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_beta_scatter(true_betas, fed_coef, ax):
    """Figure 3 equivalent: true PRS effect sizes vs federated coefficients."""
    r, p = pearsonr(true_betas, fed_coef)
    rmse = np.sqrt(np.mean((true_betas - fed_coef) ** 2))
    mask = np.abs(true_betas) > 1e-8
    mape = np.mean(np.abs((true_betas[mask] - fed_coef[mask]) / true_betas[mask])) * 100

    ax.scatter(true_betas, fed_coef, alpha=0.4, s=18,
               color=C["federated"], edgecolors="none")

    lim = max(np.abs(true_betas).max(), np.abs(fed_coef).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.5, label="Perfect recovery")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(
        f"Figure 3 – True PRS Effect Sizes vs\nFederated Estimated Coefficients",
        fontweight="bold")
    ax.set_xlabel("True PRS Beta (effect size)")
    ax.set_ylabel("Federated Estimated Coefficient")
    textstr = (f"Pearson r = {r:.4f}\n"
               f"p = {p:.2e}\n"
               f"RMSE = {rmse:.5f}\n"
               f"MAPE = {mape:.1f}%\n"
               f"N variants = {len(true_betas)}")
    ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    ax.legend(fontsize=8); sns.despine(ax=ax)


def _plot_metrics_bar(dfs, fed, local_models, ax):
    """
    Bar chart: final-round per-cohort AUC for local-only vs federated model.
    """
    x      = np.arange(len(NAMES))
    width  = 0.35
    local_auc = []
    fed_auc   = []
    for name in NAMES:
        # Local model AUC on its own test set
        probs_l = expit(dfs[name]["X_te"] @ local_models[name].coef_[0]
                        + local_models[name].intercept_[0])
        local_auc.append(roc_auc_score(dfs[name]["y_te"], probs_l))
        # Federated AUC on the same test set
        probs_f = expit(dfs[name]["X_te"] @ fed["global_coef"] + fed["global_int"][0])
        fed_auc.append(roc_auc_score(dfs[name]["y_te"], probs_f))

    bars1 = ax.bar(x - width/2, local_auc, width, label="Local-only model",
                   color=[C[n] for n in NAMES], alpha=0.45, edgecolor="white")
    bars2 = ax.bar(x + width/2, fed_auc,   width, label="Federated model",
                   color=[C[n] for n in NAMES], alpha=0.9, edgecolor="white")
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                f"{h:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0.5, color="grey", linewidth=1, linestyle=":", label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[n] for n in NAMES], fontsize=8)
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.45, max(fed_auc + local_auc) * 1.12)
    ax.set_title("Local-Only vs Federated AUC per Cohort", fontweight="bold")
    ax.legend(fontsize=8); sns.despine(ax=ax)


def _plot_final_metrics_table(dfs, fed, ax):
    """Text table of the documented performance metrics."""
    X_te_all = np.vstack([dfs[n]["X_te"] for n in NAMES])
    y_te_all  = np.concatenate([dfs[n]["y_te"] for n in NAMES])
    probs = expit(X_te_all @ fed["global_coef"] + fed["global_int"][0])
    preds = (probs >= 0.5).astype(int)

    metrics = [
        ("Test Accuracy",  f"{accuracy_score(y_te_all, preds):.4f}",   "0.5970"),
        ("Test AUC",       f"{roc_auc_score(y_te_all, probs):.4f}",    "0.6396"),
        ("Log Loss",       f"{log_loss(y_te_all, probs):.4f}",         "0.6741"),
        ("Precision",      f"{precision_score(y_te_all, preds, zero_division=0):.4f}", "0.6232"),
        ("Recall",         f"{recall_score(y_te_all, preds):.4f}",     "0.4909"),
        ("F1 Score",       f"{f1_score(y_te_all, preds):.4f}",         "0.5492"),
    ]
    ax.axis("off")
    col_labels = ["Metric", "This Run", "Documented"]
    table = ax.table(
        cellText   = metrics,
        colLabels  = col_labels,
        loc        = "center",
        cellLoc    = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    # Header styling
    for j in range(3):
        table[(0, j)].set_facecolor("#374151")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    # Row colouring
    for i in range(1, len(metrics) + 1):
        bg = "#F9FAFB" if i % 2 == 0 else "white"
        for j in range(3):
            table[(i, j)].set_facecolor(bg)

    ax.set_title("Combined Test Set — Classification Metrics\n(vs Documented Scenario)",
                 fontweight="bold", pad=20)


# ── Master generator ──────────────────────────────────────────────────────────

def generate_report(
    data_dir    : str  = ".",
    output_dir  : str  = "results",
    cohort_metas: list = None,
    n_rounds    : int  = 10,
) -> list:
    os.makedirs(output_dir, exist_ok=True)

    print("  [plot] Loading datasets …")
    dfs, true_betas = _load(data_dir)

    print(f"  [plot] Simulating federation ({n_rounds} rounds) …")
    fed = _simulate_federation(dfs, n_rounds=n_rounds, local_epochs=5)

    print("  [plot] Fitting local models …")
    local_m = _local_models(dfs)

    # Save model
    print("  [plot] Saving trained model …")
    from model_io import save_model
    from model import create_model, set_parameters as _set_params
    final_model = create_model()
    _set_params(final_model, [fed["global_coef"], fed["global_int"]])
    model_paths = save_model(
        model        = final_model,
        feature_cols = dfs[NAMES[0]]["snp_cols"],
        cohort_metas = cohort_metas or [],
        n_rounds     = n_rounds,
        output_dir   = output_dir,
        true_betas   = true_betas,
    )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    paths = list(model_paths.values())

    # ── Page 1: Cohort & mechanics ────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))
    fig1.suptitle(
        "Federated PRS Learning — Page 1: Cohort Characteristics & Federation Mechanics\n"
        "Flower FedAvg · Logistic Regression · 3 Age-Imbalanced Cohorts · 313 SNPs",
        fontsize=12, fontweight="bold")
    fig1.subplots_adjust(hspace=0.48, wspace=0.32)

    _plot_age_dist(dfs, axes1[0, 0])
    _plot_case_rate(dfs, axes1[0, 1])
    _plot_norm_convergence(fed, axes1[0, 2])
    _plot_auc_rounds(fed, axes1[1, 0])
    _plot_f1_rounds(fed, axes1[1, 1])
    _plot_ll_rounds(fed, axes1[1, 2])

    p1 = os.path.join(output_dir, "prs_federation_p1_mechanics.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [plot] Saved → {p1}")
    paths.append(p1)

    # ── Page 2: Figure 3 + metrics ────────────────────────────────────────────
    fig2 = plt.figure(figsize=(18, 12))
    fig2.suptitle(
        "Federated PRS Learning — Page 2: Coefficient Recovery & Predictive Performance\n"
        "Figure 3 Equivalent: True Effect Sizes vs Federated Coefficients",
        fontsize=12, fontweight="bold")
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.5, wspace=0.35)

    ax_beta = fig2.add_subplot(gs2[0, 0])
    ax_auc  = fig2.add_subplot(gs2[0, 1])
    ax_tbl  = fig2.add_subplot(gs2[1, 0])
    ax_coef = fig2.add_subplot(gs2[1, 1])

    _plot_beta_scatter(true_betas, fed["global_coef"], ax_beta)
    _plot_metrics_bar(dfs, fed, local_m, ax_auc)
    _plot_final_metrics_table(dfs, fed, ax_tbl)

    # Coefficient magnitude bar: top 20 SNPs by |federated coef|
    top20 = np.argsort(np.abs(fed["global_coef"]))[-20:][::-1]
    snp_labels = [dfs[NAMES[0]]["snp_cols"][i][:18] + "…"
                  if len(dfs[NAMES[0]]["snp_cols"][i]) > 18
                  else dfs[NAMES[0]]["snp_cols"][i]
                  for i in top20]
    colors = [C["federated"] if fed["global_coef"][i] > 0 else C["USA_old"] for i in top20]
    ax_coef.barh(range(20), fed["global_coef"][top20], color=colors, alpha=0.85)
    ax_coef.set_yticks(range(20))
    ax_coef.set_yticklabels(snp_labels, fontsize=6.5)
    ax_coef.axvline(0, color="black", linewidth=0.7)
    ax_coef.set_title("Top-20 SNPs by |Federated Coefficient|\n(red=positive, amber=negative effect)",
                      fontweight="bold")
    ax_coef.set_xlabel("Federated Coefficient")
    sns.despine(ax=ax_coef)

    p2 = os.path.join(output_dir, "prs_federation_p2_figure3.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [plot] Saved → {p2}")
    paths.append(p2)

    return paths
