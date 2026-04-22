"""
client.py
---------
Defines the Flower NumPy client used by each cohort node.

Each client:
  1. Receives the current global model parameters from the server.
  2. Trains locally for LOCAL_EPOCHS using its own age-imbalanced cohort.
  3. Sends updated parameters back to the server for FedAvg aggregation.
  4. Evaluates the global model on its local held-out test set.
"""

import numpy as np
import flwr as fl

from model import create_model, get_parameters, set_parameters, evaluate_model

LOCAL_EPOCHS = 5          # local SGD passes per federation round


class PRSClient(fl.client.NumPyClient):
    """
    A Flower client representing one age-cohort data silo.

    Parameters
    ----------
    client_id   : int    – index used for logging
    cohort_label: str    – human-readable label (e.g. 'young', 'normal', 'old')
    X_train, y_train     – local training data (numpy, already scaled)
    X_test,  y_test      – local test data
    n_features  : int    – number of input features (SNPs + age)
    """

    def __init__(
        self,
        client_id    : int,
        cohort_label : str,
        X_train      : np.ndarray,
        y_train      : np.ndarray,
        X_test       : np.ndarray,
        y_test       : np.ndarray,
        n_features   : int,
    ):
        self.client_id    = client_id
        self.cohort_label = cohort_label
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.n_features   = n_features
        self.model        = create_model(n_features)

    # ── Flower interface ────────────────────────────────────────────────────

    def get_parameters(self, config):
        """Return current local parameters to the server."""
        params = get_parameters(self.model)
        if not params:
            # First call before any fit – return zeros
            return [np.zeros(self.n_features), np.zeros(1)]
        return params

    def fit(self, parameters, config):
        """
        1. Set global parameters received from server.
        2. Run LOCAL_EPOCHS of partial_fit on local data.
        3. Return updated parameters and training metrics.
        """
        set_parameters(self.model, parameters)

        # Warm-start training: run multiple partial_fit passes
        for _ in range(LOCAL_EPOCHS):
            self.model.partial_fit(self.X_train, self.y_train)

        updated_params = get_parameters(self.model)
        train_metrics  = evaluate_model(self.model, self.X_train, self.y_train)

        num_examples = len(self.X_train)
        return (
            updated_params,
            num_examples,
            {
                "train_mse": train_metrics["mse"],
                "train_r2" : train_metrics["r2"],
                "cohort"   : self.cohort_label,
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the global model on this client's local test set."""
        set_parameters(self.model, parameters)
        metrics = evaluate_model(self.model, self.X_test, self.y_test)

        return (
            metrics["mse"],           # Flower expects loss as first return value
            len(self.X_test),
            {
                "test_mse"  : metrics["mse"],
                "test_r2"   : metrics["r2"],
                "cohort"    : self.cohort_label,
            },
        )


def make_client_fn(datasets: list, cohort_labels: list[str]):
    """
    Factory that returns a client_fn suitable for fl.simulation.start_simulation.

    datasets      – list of (X_train, X_test, y_train, y_test, scaler, meta)
    cohort_labels – list of string labels aligned with datasets
    """
    def client_fn(cid: str) -> PRSClient:
        idx          = int(cid)
        X_train, X_test, y_train, y_test, _, meta = datasets[idx]
        n_features   = X_train.shape[1]
        return PRSClient(
            client_id    = idx,
            cohort_label = cohort_labels[idx],
            X_train      = X_train,
            y_train      = y_train,
            X_test       = X_test,
            y_test       = y_test,
            n_features   = n_features,
        )

    return client_fn
