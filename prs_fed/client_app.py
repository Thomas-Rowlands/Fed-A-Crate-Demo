"""
client_app.py — TRE-side application.

Under the Message API, a ClientApp is no longer a NumPyClient subclass.
Instead, `app.train` and `app.evaluate` decorate plain functions that
receive a Message and return a Message. Each function:

    1. Reads `msg.content["arrays"]` for the global model weights.
    2. Reads `msg.content["config"]` for any per-round hyperparameters.
    3. Does its local training or evaluation.
    4. Builds a reply Message containing an ArrayRecord (for train only)
       and a MetricRecord.

`Context.node_config` carries the configuration that was passed to this
SuperNode at startup (e.g. via `--node-config "cohort-csv=/data/USA_young.csv
cohort-label=USA_young tre-num=1"`). That's where we get the local CSV path
and TRE identity from.
"""

from flwr.app       import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from prs_fed.task import (
    create_model, set_parameters, evaluate_model, load_local_cohort,
)


# ── Per-process cache ─────────────────────────────────────────────────────────
# Loading the CSV is expensive (~60k rows × 313 SNPs). The SuperNode keeps
# this Python process alive across rounds, so we cache the loaded arrays
# on first use and re-use them every round.

_local_data: dict | None = None


def _get_local_data(context: Context) -> dict:
    """Load (and cache) this TRE's cohort. Triggered on first call."""
    global _local_data
    if _local_data is not None:
        return _local_data

    csv_path     = str(context.node_config.get("cohort-csv", "/data/cohort.csv"))
    cohort_label = str(context.node_config.get("cohort-label", "cohort"))
    tre_num      = int(context.node_config.get("tre-num", 0))

    X_train, X_test, y_train, y_test, _scaler, meta = load_local_cohort(csv_path)

    _local_data = dict(
        X_train      = X_train,
        X_test       = X_test,
        y_train      = y_train,
        y_test       = y_test,
        cohort_label = cohort_label,
        tre_num      = tre_num,
        meta         = meta,
    )
    return _local_data


# ── The ClientApp ─────────────────────────────────────────────────────────────

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Local training step. Runs on the TRE for one federation round."""
    data         = _get_local_data(context)
    local_epochs = int(msg.content["config"].get("local-epochs", 5))

    # Receive the global model weights from the server
    arrays = msg.content["arrays"]
    coef, intercept = arrays.to_numpy_ndarrays()

    # Initialise local model with the global weights
    model = create_model()
    set_parameters(model, [coef, intercept])

    # Train locally
    for _ in range(local_epochs):
        model.partial_fit(data["X_train"], data["y_train"], classes=[0, 1])

    # Build reply: updated weights + training metrics
    train_metrics = evaluate_model(model, data["X_train"], data["y_train"])

    updated_arrays = ArrayRecord([model.coef_[0], model.intercept_])
    metrics = MetricRecord({
        # `num-examples` is the default key FedAvg looks at when computing the
        # sample-weighted average — keep that name unless you also change the
        # `weighted_by_key` constructor arg on the server side.
        "num-examples": data["meta"]["n_train"],
        "train-auc"   : train_metrics["auc"],
        "train-f1"    : train_metrics["f1"],
        "train-acc"   : train_metrics["accuracy"],
        "tre-num"     : data["tre_num"],
    })

    content = RecordDict({"arrays": updated_arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Local evaluation step. Score the global model on this TRE's test set."""
    data      = _get_local_data(context)
    threshold = float(msg.content["config"].get("threshold", 0.5))

    arrays = msg.content["arrays"]
    coef, intercept = arrays.to_numpy_ndarrays()

    model = create_model()
    set_parameters(model, [coef, intercept])

    m = evaluate_model(model, data["X_test"], data["y_test"], threshold=threshold)

    metrics = MetricRecord({
        "num-examples": data["meta"]["n_test"],
        "test-auc"    : m["auc"],
        "test-acc"    : m["accuracy"],
        "test-f1"     : m["f1"],
        "test-prec"   : m["precision"],
        "test-rec"    : m["recall"],
        "test-ll"     : m["log_loss_val"],
        "tre-num"     : data["tre_num"],
    })

    # An evaluate reply only needs metrics — no weights to return.
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
