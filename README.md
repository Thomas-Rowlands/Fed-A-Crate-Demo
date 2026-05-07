# PRS Federated Learning Demo

### Flower (flwr) · Linear Regression · Age-Imbalanced Genetics Cohorts

---

## Overview

A demonstration of [Flower](https://flower.ai) federated learning applied to
polygenic risk score (PRS) case/control prediction across three age-imbalanced
Trusted Research Environments (TREs).

The demo shows how three institutions, each holding sensitive genetics data
that **cannot leave their TRE**, can collaboratively train a single shared
classifier — only model weights cross the boundary, never patient data.

---

## Project Structure

```
Fed-A-Crate-Demo/
├── app/
│   ├── main.py          ← entry point — clean TRE-narrative output
│   ├── tre_logger.py    ← console output formatter (banners, tables, colours)
│   ├── data_utils.py    ← CSV loading, preprocessing, β recovery
│   ├── model.py         ← SGDClassifier (logistic regression, balanced classes)
│   ├── client.py        ← Flower NumPyClient (one per TRE)
│   ├── server.py        ← Custom FedAvg strategy with TRE narrative
│   ├── plotting.py      ← Two-page result figures
│   └── model_io.py      ← Save/load trained model + metadata
├── data/
│   ├── USA_young.csv    ← TRE 1 — Young cohort  (μ age 43)
│   ├── USA_old.csv      ← TRE 2 — Old cohort    (μ age 61)
│   └── USA_normal.csv   ← TRE 3 — Normal cohort (30–80 y)
└── results/             ← Generated outputs (model + plots + metadata)
```

---

## Installation

UV:

```bash
uv sync
```

Docker:

```bash
docker build -t fed-a-crate-demo .
docker run -it --rm fed-a-crate-demo
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

## Outputs

In the `results/` directory:

- **`prs_global_model.pkl`** — sklearn `SGDClassifier` ready for `.predict_proba()`
- **`prs_global_weights.npz`** — raw numpy weights (framework-agnostic)
- **`prs_model_metadata.json`** — feature list, federation config, coefficient recovery stats
- **`prs_federation_p1_mechanics.png`** — cohort characteristics + federation convergence
- **`prs_federation_p2_figure3.png`** — Figure 3 (true betas vs federated coefs) + metrics

---

## Loading the trained model

```python
from app.model_io import load_model
model, meta = load_model("results/")
# X_new must be standardised (StandardScaler) the same way as training data
probs = model.predict_proba(X_new_scaled)[:, 1]
preds = (probs >= 0.5).astype(int)