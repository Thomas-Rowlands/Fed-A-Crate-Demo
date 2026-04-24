"""
client.py
---------
Flower NumPyClient for federated logistic regression (case/control prediction).

Each client:
  1. Receives global parameters from server
  2. Runs LOCAL_EPOCHS of partial_fit (SGD) on its local imbalanced cohort
  3. Returns updated parameters + classification metrics
"""

import numpy as np
import flwr as fl

from model import create_model, get_parameters, set_parameters, evaluate_model

LOCAL_EPOCHS = 5


class PRSClient(fl.client.NumPyClient):

    def __init__(self, client_id, cohort_label,
                 X_train, y_train, X_test, y_test, n_features):
        self.client_id    = client_id
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
        updated = get_parameters(self.model)
        metrics = evaluate_model(self.model, self.X_train, self.y_train)
        return (
            updated,
            len(self.X_train),
            {
                "cohort"      : self.cohort_label,
                "train_auc"   : metrics["auc"],
                "train_f1"    : metrics["f1"],
                "train_acc"   : metrics["accuracy"],
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        return (
            metrics["log_loss_val"],
            len(self.X_test),
            {
                "cohort"    : self.cohort_label,
                "test_auc"  : metrics["auc"],
                "test_acc"  : metrics["accuracy"],
                "test_f1"   : metrics["f1"],
                "test_prec" : metrics["precision"],
                "test_rec"  : metrics["recall"],
                "test_ll"   : metrics["log_loss_val"],
            },
        )


def make_client_fn(datasets, cohort_labels):
    def client_fn(cid: str) -> PRSClient:
        idx = int(cid)
        X_train, X_test, y_train, y_test, _, meta = datasets[idx]
        return PRSClient(
            client_id    = idx,
            cohort_label = cohort_labels[idx],
            X_train      = X_train,
            y_train      = y_train,
            X_test       = X_test,
            y_test       = y_test,
            n_features   = X_train.shape[1],
        )
    return client_fn
