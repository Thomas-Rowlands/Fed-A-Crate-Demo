"""
plotting.py
-----------
All visualisation for the PRS federated learning experiment.

Generates a two-page multi-panel figure saved to:
  results/prs_federation_results_p1.png  (federation mechanics)
  results/prs_federation_results_p2.png  (model quality & biology)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ── Palette ──────────────────────────────────────────────────────────────────
C = {
    "USA_young"  : "#3B82F6",   # blue
    "USA_normal" : "#10B981",   # green
    "USA_old"    : "#F59E0B",   # amber
    "federated"  : "#EF4444",   # red
}
DISPLAY = {
    "USA_young"  : "Young (μ=43 y)",
    "USA_normal" : "Normal (μ=53 y)",
    "USA_old"    : "Old (μ=61 y)",
    "federated"  : "Federated (global)",
}
NAMES = ["USA_young", "USA_normal", "USA_old"]


# ── Data helpers (self-contained so plotting has no import dependency) ───────

def _load(data_dir: str):
    dfs = {}
    for name in NAMES:
        df = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
        snp_cols  = [c for c in df.columns if ":" in c]
        feat_cols = snp_cols + ["ageOfEntry"]
        X  = df[feat_cols].values.astype(np.float64)
        y  = df["prs"].values.astype(np.float64)
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        sp = int(len(Xs) * 0.8)
        dfs[name] = dict(
            X_train=Xs[:sp], X_test=Xs[sp:],
            y_train=y[:sp],  y_test=y[sp:],
            scaler=sc, df=df, snp_cols=snp_cols, feat_cols=feat_cols
        )
    return dfs


def _simulate_federation(dfs, n_rounds=10, local_epochs=5):
    """Re-run the federation so plotting.py is fully self-contained."""
    n_features     = dfs[NAMES[0]]["X_train"].shape[1]
    global_coef    = np.zeros(n_features)
    global_int     = np.zeros(1)
    prev_coef      = global_coef.copy()

    round_metrics  = {n: {"mse": [], "r2": []} for n in NAMES}
    global_mse_log = []
    norm_changes   = []
    coef_history   = []   # (n_rounds, n_features) – track convergence

    for _ in range(n_rounds):
        new_coefs, new_ints = [], []
        for name in NAMES:
            m = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4,
                             learning_rate="invscaling", eta0=0.01,
                             max_iter=1, warm_start=True, random_state=42)
            m.coef_      = global_coef.copy()
            m.intercept_ = global_int.copy()
            for _ in range(local_epochs):
                m.partial_fit(dfs[name]["X_train"], dfs[name]["y_train"])
            new_coefs.append(m.coef_.copy())
            new_ints.append(m.intercept_.copy())

        global_coef = np.mean(new_coefs, axis=0)
        global_int  = np.mean(new_ints,  axis=0)

        norm_changes.append(np.linalg.norm(global_coef - prev_coef))
        prev_coef = global_coef.copy()
        coef_history.append(global_coef.copy())

        total = 0
        for name in NAMES:
            yp  = dfs[name]["X_test"] @ global_coef + global_int[0]
            mse = float(mean_squared_error(dfs[name]["y_test"], yp))
            r2  = float(r2_score(dfs[name]["y_test"], yp))
            round_metrics[name]["mse"].append(mse)
            round_metrics[name]["r2"].append(r2)
            total += mse
        global_mse_log.append(total / 3)

    return dict(
        global_coef    = global_coef,
        global_int     = global_int,
        round_metrics  = round_metrics,
        global_mse_log = global_mse_log,
        norm_changes   = norm_changes,
        coef_history   = np.array(coef_history),
    )


def _local_models(dfs):
    models = {}
    for name in NAMES:
        models[name] = Ridge(alpha=1e-4).fit(dfs[name]["X_train"], dfs[name]["y_train"])
    return models


# ── Individual plot functions ────────────────────────────────────────────────

def _plot_age_dist(dfs, ax):
    for name in NAMES:
        ages = dfs[name]["df"]["ageOfEntry"].values
        ax.hist(ages, bins=40, alpha=0.45, color=C[name],
                label=DISPLAY[name], density=True, edgecolor="none")
        ax.axvline(ages.mean(), color=C[name], linewidth=2, linestyle="--")
    ax.set_title("Age-of-Entry Distribution\n(Intentional Cohort Imbalance)", fontweight="bold")
    ax.set_xlabel("Age of Entry"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); sns.despine(ax=ax)


def _plot_prs_dist(dfs, ax):
    for name in NAMES:
        prs = dfs[name]["df"]["prs"].values
        ax.hist(prs, bins=60, alpha=0.45, color=C[name],
                label=DISPLAY[name], density=True, edgecolor="none")
    ax.set_title("PRS Distribution per Cohort", fontweight="bold")
    ax.set_xlabel("Polygenic Risk Score"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); sns.despine(ax=ax)


def _plot_case_rate(dfs, ax):
    labels = [DISPLAY[n] for n in NAMES]
    rates  = [dfs[n]["df"]["case"].mean() * 100 for n in NAMES]
    bars   = ax.bar(labels, rates, color=[C[n] for n in NAMES], alpha=0.85, edgecolor="white")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Case Rate by Age Cohort\n(Age Imbalance Impact)", fontweight="bold")
    ax.set_ylabel("Case Rate (%)"); ax.set_ylim(0, max(rates) * 1.3)
    ax.tick_params(axis="x", labelsize=8); sns.despine(ax=ax)


def _plot_param_convergence(fed, ax):
    rounds = list(range(1, len(fed["norm_changes"]) + 1))
    ax.plot(rounds, fed["norm_changes"], "o-", color=C["federated"],
            linewidth=2.5, markersize=6)
    ax.fill_between(rounds, 0, fed["norm_changes"], alpha=0.15, color=C["federated"])
    ax.set_title("Parameter Convergence\n(‖Δ global coef‖ per Round)", fontweight="bold")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("L2 Norm of Δcoef")
    ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_mse_rounds(fed, ax):
    rounds = list(range(1, len(fed["global_mse_log"]) + 1))
    for name in NAMES:
        ax.plot(rounds, fed["round_metrics"][name]["mse"],
                "o-", color=C[name], linewidth=2, markersize=4, label=DISPLAY[name])
    ax.plot(rounds, fed["global_mse_log"], "s--", color=C["federated"],
            linewidth=2.5, markersize=6, label="Global weighted MSE")
    ax.set_title("Federated MSE per Round\n(Global model on each cohort's test set)", fontweight="bold")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("MSE")
    ax.legend(fontsize=7.5); ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_r2_rounds(fed, ax):
    rounds = list(range(1, len(fed["global_mse_log"]) + 1))
    for name in NAMES:
        ax.plot(rounds, fed["round_metrics"][name]["r2"],
                "o-", color=C[name], linewidth=2, markersize=4, label=DISPLAY[name])
    ax.set_title("Federated R² per Round\n(Global model on each cohort's test set)", fontweight="bold")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("R²")
    ax.legend(fontsize=8); ax.set_xticks(rounds); sns.despine(ax=ax)


def _plot_cross_cohort_rmse(dfs, local_models, fed, ax):
    """
    Heatmap: RMSE[train_cohort × test_cohort] for local models + federated.
    Lower = better cross-cohort generalisation.
    """
    all_keys   = NAMES + ["federated"]
    matrix     = np.zeros((len(all_keys), len(NAMES)))

    for i, train in enumerate(NAMES):
        for j, test in enumerate(NAMES):
            yp   = local_models[train].predict(dfs[test]["X_test"])
            matrix[i, j] = np.sqrt(mean_squared_error(dfs[test]["y_test"], yp))
    for j, test in enumerate(NAMES):
        yp = dfs[test]["X_test"] @ fed["global_coef"] + fed["global_int"][0]
        matrix[3, j] = np.sqrt(mean_squared_error(dfs[test]["y_test"], yp))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="RMSE", shrink=0.8)
    ax.set_xticks(range(len(NAMES)))
    ax.set_yticks(range(len(all_keys)))
    ax.set_xticklabels([DISPLAY[n] for n in NAMES], fontsize=7.5, rotation=15, ha="right")
    ax.set_yticklabels([DISPLAY[n] for n in all_keys], fontsize=7.5)
    for i in range(len(all_keys)):
        for j in range(len(NAMES)):
            ax.text(j, i, f"{matrix[i,j]:.5f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i,j] < matrix.max()*0.6 else "white")
    ax.set_title("Cross-Cohort RMSE\n(Row = model source, Col = test cohort)", fontweight="bold")
    ax.set_xlabel("Test Cohort"); ax.set_ylabel("Model Trained On")
    # Highlight federated row
    rect = plt.Rectangle((-0.5, 2.5), len(NAMES), 1,
                          fill=False, edgecolor="#EF4444", linewidth=2.5)
    ax.add_patch(rect)


def _plot_snp_weight_diff(dfs, local_models, fed, ax):
    """
    Top 20 SNPs where young vs old model weights diverge most.
    Shows the federated weight as a dot for comparison.
    """
    feat_cols = dfs["USA_young"]["feat_cols"]
    cy  = local_models["USA_young"].coef_
    co  = local_models["USA_old"].coef_
    cf  = fed["global_coef"]

    diff    = np.abs(cy - co)
    top_idx = np.argsort(diff)[-20:][::-1]

    snp_labels = [feat_cols[i].split(":")[0] + ":" + feat_cols[i].split(":")[1]
                  if ":" in feat_cols[i] else feat_cols[i]
                  for i in top_idx]
    y_pos = np.arange(len(top_idx))

    ax.barh(y_pos,  cy[top_idx], 0.35, color=C["USA_young"], alpha=0.8,
            label="Young", align="center")
    ax.barh(y_pos - 0.35, co[top_idx], 0.35, color=C["USA_old"], alpha=0.8,
            label="Old", align="center")
    ax.scatter(cf[top_idx], y_pos - 0.175, color=C["federated"], zorder=5,
               s=30, label="Federated", marker="D")

    ax.set_yticks(y_pos - 0.175)
    ax.set_yticklabels(snp_labels, fontsize=6.5)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_title("Top-20 SNPs: Young vs Old Model Weight Divergence\n(Federated weight shown as ◆)", fontweight="bold")
    ax.set_xlabel("Standardised Coefficient")
    ax.legend(fontsize=8, loc="lower right"); sns.despine(ax=ax)


def _plot_scatter_trio(dfs, fed, local_models, axes):
    """
    For each cohort: scatter true PRS vs predicted PRS for BOTH the
    cohort's own local model (faint) and the federated model (solid).
    2000-point subsample.
    """
    rng = np.random.default_rng(42)
    for i, (name, ax) in enumerate(zip(NAMES, axes)):
        X_te = dfs[name]["X_test"]
        y_te = dfs[name]["y_test"]
        idx  = rng.choice(len(y_te), size=min(2000, len(y_te)), replace=False)

        y_local = local_models[name].predict(X_te[idx])
        y_fed   = X_te[idx] @ fed["global_coef"] + fed["global_int"][0]

        lim = [y_te[idx].min() - 0.05, y_te[idx].max() + 0.05]
        ax.scatter(y_te[idx], y_local, alpha=0.15, s=5, color=C[name],
                   label="Local model")
        ax.scatter(y_te[idx], y_fed,   alpha=0.25, s=5, color=C["federated"],
                   label="Federated model")
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5)

        mse_l = mean_squared_error(y_te[idx], y_local)
        mse_f = mean_squared_error(y_te[idx], y_fed)
        r2_f  = r2_score(y_te[idx], y_fed)

        ax.set_title(
            f"{DISPLAY[name]}\nLocal RMSE={np.sqrt(mse_l):.5f}  "
            f"Fed RMSE={np.sqrt(mse_f):.5f}  R²={r2_f:.6f}",
            fontweight="bold", fontsize=8)
        ax.set_xlabel("True PRS", fontsize=8); ax.set_ylabel("Predicted PRS", fontsize=8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=7); sns.despine(ax=ax)


# ── Master report generator ──────────────────────────────────────────────────

def generate_report(
    data_dir    : str = ".",
    output_dir  : str = "results",
    cohort_metas: list = None,
    n_rounds    : int = 10,
) -> list[str]:
    """
    Load data, run the full federation, save the trained model, build two
    rich figures.  Returns list of output paths (PNGs + model files).
    """
    os.makedirs(output_dir, exist_ok=True)
    print("  [plot] Loading datasets …")
    dfs = _load(data_dir)
    print(f"  [plot] Simulating federation ({n_rounds} rounds) …")
    fed = _simulate_federation(dfs, n_rounds=n_rounds, local_epochs=5)
    print("  [plot] Fitting local models …")
    local_m = _local_models(dfs)

    # ── Persist the final global model ───────────────────────────────────────
    print("  [plot] Saving trained model …")
    from model_io import save_model
    from model import create_model, set_parameters as _set_params
    final_model = create_model(fed["global_coef"].shape[0])
    _set_params(final_model, [fed["global_coef"], fed["global_int"]])
    model_paths = save_model(
        model        = final_model,
        feature_cols = dfs[NAMES[0]]["feat_cols"],
        cohort_metas = cohort_metas or [],
        n_rounds     = n_rounds,
        output_dir   = output_dir,
    )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    paths = list(model_paths.values())

    # ── Page 1: Federation mechanics ─────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))
    fig1.suptitle(
        "Federated PRS Learning — Page 1: Cohort Characteristics & Federation Mechanics\n"
        "Flower FedAvg · SGDRegressor · 3 Age-Imbalanced Cohorts · 313 SNPs + Age",
        fontsize=12, fontweight="bold")
    fig1.subplots_adjust(hspace=0.48, wspace=0.32)

    _plot_age_dist(dfs, axes1[0, 0])
    _plot_prs_dist(dfs, axes1[0, 1])
    _plot_case_rate(dfs, axes1[0, 2])
    _plot_param_convergence(fed, axes1[1, 0])
    _plot_mse_rounds(fed, axes1[1, 1])
    _plot_r2_rounds(fed, axes1[1, 2])

    p1 = os.path.join(output_dir, "prs_federation_p1_mechanics.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [plot] Saved → {p1}")
    paths.append(p1)

    # ── Page 2: Model quality & biology ──────────────────────────────────────
    fig2 = plt.figure(figsize=(18, 14))
    fig2.suptitle(
        "Federated PRS Learning — Page 2: Model Quality & Biological Insights\n"
        "Cross-Cohort Generalisation · SNP Weight Divergence · Prediction Scatter",
        fontsize=12, fontweight="bold")
    gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.48, wspace=0.35)

    ax_hm   = fig2.add_subplot(gs2[0, :2])   # wide heatmap
    ax_snp  = fig2.add_subplot(gs2[0, 2])    # SNP weights
    ax_sc   = [fig2.add_subplot(gs2[1, j]) for j in range(3)]

    _plot_cross_cohort_rmse(dfs, local_m, fed, ax_hm)
    _plot_snp_weight_diff(dfs, local_m, fed, ax_snp)
    _plot_scatter_trio(dfs, fed, local_m, ax_sc)

    p2 = os.path.join(output_dir, "prs_federation_p2_quality.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [plot] Saved → {p2}")
    paths.append(p2)

    return paths
