"""
plotting.py — Federated PRS demo visualisation.
Page 1: cohort characteristics + federation mechanics
Page 2: Figure 3 (true betas vs federated coefs) + classification metrics
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
    "USA_old"   : "#F59E0B",
    "USA_normal": "#10B981",
    "federated" : "#EF4444",
}
DISPLAY = {
    "USA_young" : "TRE 1 – Young (μ=43 y)",
    "USA_old"   : "TRE 2 – Old   (μ=61 y)",
    "USA_normal": "TRE 3 – Normal (30–80 y)",
    "federated" : "Federated (global)",
}
NAMES = ["USA_young", "USA_old", "USA_normal"]


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
    ref = dfs["USA_normal"]
    betas = Ridge(alpha=1e-10).fit(
        ref["df"][ref["snp_cols"]].values.astype(np.float64),
        ref["df"]["prs"].values.astype(np.float64)
    ).coef_
    return dfs, betas


def _simulate_federation(dfs, n_rounds=10, local_epochs=5):
    np.random.seed(42)
    n_features  = dfs[NAMES[0]]["X_tr"].shape[1]
    global_coef = np.zeros(n_features)
    global_int  = np.zeros(1)
    prev_coef   = global_coef.copy()
    rm = {n: {"auc":[], "f1":[], "acc":[], "ll":[]} for n in NAMES}
    global_ll = []; norm_changes = []

    for _ in range(n_rounds):
        new_coefs, new_ints = [], []
        for name in NAMES:
            m = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                              learning_rate="invscaling", eta0=0.01,
                              max_iter=1, warm_start=True, random_state=42,
                              class_weight="balanced")
            m.coef_      = global_coef.reshape(1,-1).copy()
            m.intercept_ = global_int.copy()
            m.classes_   = np.array([0,1])
            for _ in range(local_epochs):
                m.partial_fit(dfs[name]["X_tr"], dfs[name]["y_tr"], classes=[0,1])
            new_coefs.append(m.coef_[0].copy())
            new_ints.append(m.intercept_.copy())
        global_coef = np.mean(new_coefs, axis=0)
        global_int  = np.mean(new_ints,  axis=0)
        norm_changes.append(np.linalg.norm(global_coef - prev_coef))
        prev_coef = global_coef.copy()

        total = 0
        for name in NAMES:
            probs = expit(dfs[name]["X_te"] @ global_coef + global_int[0])
            preds = (probs >= 0.5).astype(int)
            y_te  = dfs[name]["y_te"]
            rm[name]["auc"].append(float(roc_auc_score(y_te, probs)))
            rm[name]["f1"].append(float(f1_score(y_te, preds, average="macro", zero_division=0)))
            rm[name]["acc"].append(float(accuracy_score(y_te, preds)))
            ll = float(log_loss(y_te, probs))
            rm[name]["ll"].append(ll); total += ll
        global_ll.append(total / 3)

    return dict(global_coef=global_coef, global_int=global_int,
                round_metrics=rm, global_ll_log=global_ll,
                norm_changes=norm_changes)


def _local_models(dfs):
    out = {}
    for name in NAMES:
        m = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          learning_rate="invscaling", eta0=0.01,
                          max_iter=1, warm_start=True, random_state=42,
                          class_weight="balanced")
        m.classes_ = np.array([0,1])
        for _ in range(10):
            m.partial_fit(dfs[name]["X_tr"], dfs[name]["y_tr"], classes=[0,1])
        out[name] = m
    return out


# ── Plot panels ─────────────────────────────────────────────────────────────

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
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{rate:.2f}%\n(n={cnt})", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_title("Case Rate per TRE", fontweight="bold")
    ax.set_ylabel("Case Rate (%)"); ax.set_ylim(0, max(rates)*1.5)
    ax.tick_params(axis="x", labelsize=7.5); sns.despine(ax=ax)


def _plot_norm(fed, ax):
    rounds = list(range(1, len(fed["norm_changes"])+1))
    ax.plot(rounds, fed["norm_changes"], "o-", color=C["federated"], linewidth=2.5, markersize=6)
    ax.fill_between(rounds, 0, fed["norm_changes"], alpha=0.15, color=C["federated"])
    ax.set_title("Parameter Convergence\n‖Δ global coef‖ per Round", fontweight="bold")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("L2 Norm of Δcoef")
    ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_auc(fed, ax):
    rounds = list(range(1, len(fed["global_ll_log"])+1))
    for name in NAMES:
        ax.plot(rounds, fed["round_metrics"][name]["auc"], "o-",
                color=C[name], linewidth=2, markersize=4, label=DISPLAY[name])
    ax.axhline(0.5, color="grey", linewidth=1, linestyle=":", label="Random baseline")
    ax.set_title("AUC per Federation Round", fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=7); ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_f1(fed, ax):
    rounds = list(range(1, len(fed["global_ll_log"])+1))
    for name in NAMES:
        ax.plot(rounds, fed["round_metrics"][name]["f1"], "o-",
                color=C[name], linewidth=2, markersize=4, label=DISPLAY[name])
    ax.set_title("F1 per Federation Round", fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel("F1 Score")
    ax.legend(fontsize=7); ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_ll(fed, ax):
    rounds = list(range(1, len(fed["global_ll_log"])+1))
    for name in NAMES:
        ax.plot(rounds, fed["round_metrics"][name]["ll"], "o-",
                color=C[name], linewidth=2, markersize=4, label=DISPLAY[name])
    ax.plot(rounds, fed["global_ll_log"], "s--", color=C["federated"],
            linewidth=2.5, markersize=6, label="Global avg")
    ax.set_title("Log-Loss per Federation Round", fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel("Log Loss")
    ax.legend(fontsize=7); ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_beta(true_betas, fed_coef, ax):
    r, p = pearsonr(true_betas, fed_coef)
    rmse = np.sqrt(np.mean((true_betas - fed_coef)**2))
    mask = np.abs(true_betas) > 1e-8
    mape = np.mean(np.abs((true_betas[mask] - fed_coef[mask]) / true_betas[mask]))*100
    ax.scatter(true_betas, fed_coef, alpha=0.4, s=18,
               color=C["federated"], edgecolors="none")
    lim = max(np.abs(true_betas).max(), np.abs(fed_coef).max()) * 1.1
    ax.plot([-lim,lim], [-lim,lim], "k--", linewidth=1, alpha=0.5, label="Perfect recovery")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_title("Figure 3 – True PRS Effect Sizes vs\nFederated Estimated Coefficients",
                 fontweight="bold")
    ax.set_xlabel("True PRS Beta (effect size)")
    ax.set_ylabel("Federated Estimated Coefficient")
    txt = (f"Pearson r = {r:.4f}\np = {p:.2e}\nRMSE = {rmse:.5f}\n"
           f"MAPE = {mape:.1f}%\nN variants = {len(true_betas)}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    ax.legend(fontsize=8); sns.despine(ax=ax)


def _plot_metrics_bar(dfs, fed, local_models, ax):
    x = np.arange(len(NAMES)); w = 0.35
    local_auc, fed_auc = [], []
    for name in NAMES:
        probs_l = expit(dfs[name]["X_te"] @ local_models[name].coef_[0]
                        + local_models[name].intercept_[0])
        local_auc.append(roc_auc_score(dfs[name]["y_te"], probs_l))
        probs_f = expit(dfs[name]["X_te"] @ fed["global_coef"] + fed["global_int"][0])
        fed_auc.append(roc_auc_score(dfs[name]["y_te"], probs_f))
    b1 = ax.bar(x-w/2, local_auc, w, label="Local-only model",
                color=[C[n] for n in NAMES], alpha=0.45, edgecolor="white")
    b2 = ax.bar(x+w/2, fed_auc,   w, label="Federated model",
                color=[C[n] for n in NAMES], alpha=0.9,  edgecolor="white")
    for bar in list(b1)+list(b2):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.002,
                f"{h:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0.5, color="grey", linewidth=1, linestyle=":", label="Random baseline")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[n] for n in NAMES], fontsize=8)
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.45, max(fed_auc+local_auc)*1.12)
    ax.set_title("Local-Only vs Federated AUC per TRE", fontweight="bold")
    ax.legend(fontsize=8); sns.despine(ax=ax)


def _plot_metrics_table(dfs, fed, ax):
    X_te_all = np.vstack([dfs[n]["X_te"] for n in NAMES])
    y_te_all = np.concatenate([dfs[n]["y_te"] for n in NAMES])
    probs = expit(X_te_all @ fed["global_coef"] + fed["global_int"][0])
    preds = (probs >= 0.5).astype(int)
    rows = [
        ("Test Accuracy",  f"{accuracy_score(y_te_all, preds):.4f}"),
        ("Test AUC",       f"{roc_auc_score(y_te_all, probs):.4f}"),
        ("Log Loss",       f"{log_loss(y_te_all, probs):.4f}"),
        ("Precision",      f"{precision_score(y_te_all, preds, average='macro', zero_division=0):.4f}"),
        ("Recall",         f"{recall_score(y_te_all, preds, average='macro', zero_division=0):.4f}"),
        ("F1 Score",       f"{f1_score(y_te_all, preds, average='macro', zero_division=0):.4f}"),
    ]
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Metric","Value"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.0, 2.0)
    for j in range(2):
        tbl[(0,j)].set_facecolor("#374151")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        bg = "#F9FAFB" if i % 2 == 0 else "white"
        for j in range(2):
            tbl[(i,j)].set_facecolor(bg)
    ax.set_title("Combined Test Set — Classification Metrics",
                 fontweight="bold", pad=20)


def generate_report(data_dir=".", output_dir="results",
                    cohort_metas=None, n_rounds=10):
    os.makedirs(output_dir, exist_ok=True)
    dfs, true_betas = _load(data_dir)
    fed = _simulate_federation(dfs, n_rounds=n_rounds, local_epochs=5)
    local_m = _local_models(dfs)

    # Save model
    from model_io import save_model
    from model    import create_model, set_parameters
    final_model = create_model()
    set_parameters(final_model, [fed["global_coef"], fed["global_int"]])
    model_paths = save_model(final_model, dfs[NAMES[0]]["snp_cols"],
                             cohort_metas or [], n_rounds, output_dir, true_betas)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    paths = list(model_paths.values())

    # Page 1
    fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig1.suptitle("Federated PRS Learning — Page 1: Cohort Characteristics & Federation Mechanics\n"
                  "Flower FedAvg · Logistic Regression · 3 Age-Imbalanced TREs · 313 SNPs",
                  fontsize=12, fontweight="bold")
    fig1.subplots_adjust(hspace=0.48, wspace=0.32)
    _plot_age_dist(dfs, axes[0,0]); _plot_case_rate(dfs, axes[0,1]); _plot_norm(fed, axes[0,2])
    _plot_auc(fed, axes[1,0]);     _plot_f1(fed, axes[1,1]);        _plot_ll(fed, axes[1,2])
    p1 = os.path.join(output_dir, "prs_federation_p1_mechanics.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig1)
    paths.append(p1)

    # Page 2
    fig2 = plt.figure(figsize=(18, 12))
    fig2.suptitle("Federated PRS Learning — Page 2: Coefficient Recovery & Predictive Performance",
                  fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.5, wspace=0.35)
    ax_b  = fig2.add_subplot(gs[0,0]); ax_a = fig2.add_subplot(gs[0,1])
    ax_t  = fig2.add_subplot(gs[1,0]); ax_c = fig2.add_subplot(gs[1,1])
    _plot_beta(true_betas, fed["global_coef"], ax_b)
    _plot_metrics_bar(dfs, fed, local_m, ax_a)
    _plot_metrics_table(dfs, fed, ax_t)

    # Top-20 SNP coefs
    top20 = np.argsort(np.abs(fed["global_coef"]))[-20:][::-1]
    snp_labels = [dfs[NAMES[0]]["snp_cols"][i][:18] + "…"
                  if len(dfs[NAMES[0]]["snp_cols"][i]) > 18
                  else dfs[NAMES[0]]["snp_cols"][i] for i in top20]
    colors = [C["federated"] if fed["global_coef"][i] > 0 else C["USA_old"] for i in top20]
    ax_c.barh(range(20), fed["global_coef"][top20], color=colors, alpha=0.85)
    ax_c.set_yticks(range(20)); ax_c.set_yticklabels(snp_labels, fontsize=6.5)
    ax_c.axvline(0, color="black", linewidth=0.7)
    ax_c.set_title("Top-20 SNPs by |Federated Coefficient|\n(red=positive, amber=negative)",
                   fontweight="bold")
    ax_c.set_xlabel("Federated Coefficient"); sns.despine(ax=ax_c)

    p2 = os.path.join(output_dir, "prs_federation_p2_figure3.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig2)
    paths.append(p2)
    return paths
