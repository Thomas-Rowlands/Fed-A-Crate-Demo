"""
client_main.py — Entry point for a single TRE client.

Run inside each TRE (or each Docker container). Loads ONLY that TRE's
local cohort CSV, connects to the central Flower server, and participates
in federated training rounds. Patient data never leaves this process —
only model parameters are sent to the server.

Usage:
    python client_main.py \\
        --server      fed-server:8080 \\
        --cohort-csv  /data/cohort.csv \\
        --cohort-label USA_young \\
        --tre-num     1
        [--certs-dir  /certs]

Environment variables (used as defaults if CLI args not given):
    FLWR_SERVER     equivalent to --server
    COHORT_CSV      equivalent to --cohort-csv
    COHORT_LABEL    equivalent to --cohort-label
    TRE_NUM         equivalent to --tre-num
    CERTS_DIR       equivalent to --certs-dir
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import flwr as fl

from data_utils import load_local_cohort
from model      import (create_model, get_parameters, set_parameters,
                        evaluate_model)
from tre_logger import (
    silence_third_party_logs, banner,
    client_waiting, client_cohort_summary,
    client_round_received, client_training,
    client_returning, client_evaluated, done,
)


LOCAL_EPOCHS = 5


# ── The actual Flower client ────────────────────────────────────────────────

class TREClient(fl.client.NumPyClient):

    def __init__(self, tre_num, cohort_label,
                 X_train, y_train, X_test, y_test, n_features):
        self.tre_num      = tre_num
        self.cohort_label = cohort_label
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.n_features   = n_features
        self.model        = create_model()

    def get_parameters(self, config):
        params = get_parameters(self.model)
        if not params:
            return [np.zeros(self.n_features), np.zeros(1)]
        return params

    def fit(self, parameters, config):
        client_round_received(self.tre_num, self.cohort_label)
        set_parameters(self.model, parameters)

        client_training(self.tre_num, len(self.X_train), LOCAL_EPOCHS)
        for _ in range(LOCAL_EPOCHS):
            self.model.partial_fit(self.X_train, self.y_train, classes=[0, 1])

        train_metrics = evaluate_model(self.model, self.X_train, self.y_train)
        client_returning(self.tre_num,
                         auc=train_metrics["auc"], f1=train_metrics["f1"])

        return (
            get_parameters(self.model),
            len(self.X_train),
            {
                "cohort"   : self.cohort_label,
                "tre_num"  : self.tre_num,
                "train_auc": train_metrics["auc"],
                "train_f1" : train_metrics["f1"],
                "train_acc": train_metrics["accuracy"],
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        m = evaluate_model(self.model, self.X_test, self.y_test)
        client_evaluated(self.tre_num, auc=m["auc"], f1=m["f1"], acc=m["accuracy"])
        return (
            m["log_loss_val"],
            len(self.X_test),
            {
                "cohort"   : self.cohort_label,
                "tre_num"  : self.tre_num,
                "test_auc" : m["auc"],
                "test_acc" : m["accuracy"],
                "test_f1"  : m["f1"],
                "test_prec": m["precision"],
                "test_rec" : m["recall"],
                "test_ll"  : m["log_loss_val"],
            },
        )


# ── Bootstrap ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PRS federation TRE client")
    p.add_argument("--server",        default=os.environ.get("FLWR_SERVER", "localhost:8080"))
    p.add_argument("--cohort-csv",    default=os.environ.get("COHORT_CSV", "/data/cohort.csv"))
    p.add_argument("--cohort-label",  default=os.environ.get("COHORT_LABEL", "cohort"))
    p.add_argument("--tre-num",       type=int,
                                      default=int(os.environ.get("TRE_NUM", "0")))
    p.add_argument("--certs-dir",     default=os.environ.get("CERTS_DIR", ""),
                   help="Directory containing ca.crt for TLS. Empty = no TLS.")
    p.add_argument("--retry-seconds", type=int, default=5,
                   help="If server isn't reachable yet, wait this many "
                        "seconds and retry (Docker Compose timing).")
    p.add_argument("--max-retries",   type=int, default=30,
                   help="How many connection retries before giving up.")
    return p.parse_args()


def _read_ca_cert(certs_dir: str):
    """Load CA certificate as bytes for TLS connection, or None."""
    if not certs_dir:
        return None
    ca_path = Path(certs_dir) / "ca.crt"
    if not ca_path.exists():
        print(f"  [WARN] --certs-dir set but {ca_path} not found; connecting without TLS")
        return None
    return ca_path.read_bytes()


def main():
    args = parse_args()
    silence_third_party_logs()

    banner(
        role       = f"TRE {args.tre_num} client",
        identifier = args.cohort_label,
        subtitle   = f"Connecting to {args.server}  (Flower federation)",
    )

    # ── Load local cohort ────────────────────────────────────────────────────
    if not os.path.exists(args.cohort_csv):
        print(f"  [ERROR] Cohort CSV not found: {args.cohort_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading local cohort from {args.cohort_csv} …")
    X_train, X_test, y_train, y_test, _scaler, meta = load_local_cohort(args.cohort_csv)
    client_cohort_summary(args.tre_num, meta)

    client = TREClient(
        tre_num      = args.tre_num,
        cohort_label = args.cohort_label,
        X_train      = X_train,
        y_train      = y_train,
        X_test       = X_test,
        y_test       = y_test,
        n_features   = X_train.shape[1],
    ).to_client()

    # ── Connect to the server (with retries — handy under Docker Compose) ───
    ca_cert = _read_ca_cert(args.certs_dir)
    client_waiting(args.server)

    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            fl.client.start_client(
                server_address = args.server,
                client         = client,
                root_certificates = ca_cert,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < args.max_retries:
                print(f"  [retry {attempt}/{args.max_retries}] "
                      f"server unreachable ({e.__class__.__name__}); "
                      f"waiting {args.retry_seconds}s…")
                time.sleep(args.retry_seconds)
            else:
                print(f"  [ERROR] giving up after {attempt} attempts: {last_error}",
                      file=sys.stderr)
                sys.exit(1)

    done(f"TRE {args.tre_num} finished participating in federation.")


if __name__ == "__main__":
    main()
