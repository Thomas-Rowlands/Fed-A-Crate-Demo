# PRS Federated Learning Demo
### Flower (flwr) · Linear Regression · Age-Imbalanced Genetics Cohorts

---

## Overview

This project demonstrates **federated learning for Polygenic Risk Score (PRS)
prediction** using three intentionally age-imbalanced genetics datasets.

| Client | Dataset         | Cohort      | Age μ  |
|--------|-----------------|-------------|--------|
| 0      | USA_young.csv   | Young       | ~43 y  |
| 1      | USA_normal.csv  | Normal      | ~53 y  |
| 2      | USA_old.csv     | Old         | ~61 y  |

Each dataset contains **60,000 individuals** with **313 SNP features** (dosage
0/1/2) plus age-of-entry, and a continuous **PRS** target.

The federation demonstrates how **FedAvg** aggregation produces a global model
that generalises across age groups better than any single-cohort model.

---

## Project Structure

```
prs_federation/
├── main.py          Entry point – runs the full simulation
├── data_utils.py    CSV loading, feature engineering, train/test split
├── model.py         SGDRegressor wrapper + Flower parameter serialisation
├── client.py        Flower NumPyClient (one per cohort)
├── server.py        Custom FedAvg strategy with per-round metric logging
├── plotting.py      All visualisation (6-panel results figure)
└── results/         Output directory (created at runtime)
    └── prs_federation_results.png
```

---

## Installation

```bash
pip install flwr[simulation] scikit-learn pandas numpy matplotlib seaborn
```

---

## Running

```bash
# From the prs_federation/ directory (CSVs in the same folder)
python main.py

# Or specify paths explicitly
python main.py --data-dir /path/to/csvs --rounds 10 --output-dir results
```

---

## Key Design Decisions

### Why SGDRegressor instead of LinearRegression?
`SGDRegressor` supports `partial_fit`, which lets each client run multiple
local epochs before sending updates to the server – exactly how neural-network
FL clients work.  The parameter vector (coef_ + intercept_) is directly
exchangeable as NumPy arrays, making FedAvg trivial.

### Features
- **313 SNP dosages** (0/1/2 – standard GWAS encoding)
- **Age of entry** – included as a continuous feature to help the model
  account for age-related PRS differences

### FedAvg Aggregation
Parameters are aggregated as a **weighted average** where each client's
contribution is proportional to its number of training samples.  Since all
three cohorts have the same size (60 k), the global model is an equal-weight
average in this demo – but the weighting logic is fully general.

### Age Imbalance Demonstration
- **Round 1**: Each client evaluates with near-zero federation benefit
  (only one round of global averaging).
- **Final round**: All cohorts should show improved or converged MSE as the
  global model learns patterns spanning all age groups.

---

## Output

The script produces a **6-panel figure** (`results/prs_federation_results.png`):

1. **Age Distribution** – Gaussian approximations of each cohort's age-of-entry
2. **MSE over Rounds** – Per-cohort and global weighted MSE across federation rounds
3. **R² over Rounds** – Per-cohort R² across federation rounds
4. **Local vs Federated MSE** – Bar chart comparing round-1 (local-only) vs final-round MSE
5–7. **Prediction Scatter** – True PRS vs predicted PRS for each cohort using the final global model

---

## Extending This Demo

- **Add differential privacy**: wrap the strategy with `DifferentialPrivacyClientSideFixedClipping`
- **Unequal sampling**: set `fraction_fit < 1.0` to sample only a subset of clients per round
- **Survival analysis**: replace PRS regression with a Cox proportional hazards model
- **More cohorts**: add additional CSV files and extend `cohort_labels` in `main.py`
