"""
client.py — Flower NumPyClient representing one TRE.

Each TRE:
  1. Receives the global model from the central server
  2. Trains locally for LOCAL_EPOCHS (data never leaves the TRE)
  3. Returns updated weights to the server (only the model crosses the boundary)

Narrative log messages are emitted from the SERVER side (server.py) rather
than here, because client code runs inside Ray worker processes whose stdout
gets prefixed with '(ClientAppActor pid=…)' — too noisy for a clean demo.
"""

import numpy as np
import flwr as fl
from flwr.common import Context

from model import (create_model, get_parameters,
                    set_parameters, evaluate_model)

LOCAL_EPOCHS = 5


class TREClient(fl.client.NumPyClient):

    def __init__(self, tre_num, cohort_label, X_train, y_train,
                 X_test, y_test, n_features):
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
        set_parameters(self.model, parameters)
        for _ in range(LOCAL_EPOCHS):
            self.model.partial_fit(self.X_train, self.y_train, classes=[0, 1])

        train_metrics = evaluate_model(self.model, self.X_train, self.y_train)
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


def make_client_fn(datasets, cohort_labels):
    """Modern Context-based Flower client factory."""
    def client_fn(context: Context) -> fl.client.Client:
        cid = (context.node_config.get("partition-id")
               if hasattr(context, "node_config") else None)
        if cid is None:
            cid = int(context.cid) if hasattr(context, "cid") else 0
        idx = int(cid)

        X_train, X_test, y_train, y_test, _, _ = datasets[idx]
        return TREClient(
            tre_num      = idx + 1,
            cohort_label = cohort_labels[idx],
            X_train      = X_train,
            y_train      = y_train,
            X_test       = X_test,
            y_test       = y_test,
            n_features   = X_train.shape[1],
        ).to_client()
    return client_fn
