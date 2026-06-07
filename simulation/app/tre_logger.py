"""
tre_logger.py
-------------
Clean, presentation-friendly console output for the federated TRE demo.

Replaces verbose Flower/Ray internals with a narrative that walks researchers
through what's happening at each step:
  - Sending model to TRE
  - Training at TRE
  - Receiving updated weights
  - Aggregating

Use silence_third_party_logs() at startup to suppress Flower/Ray chatter
so only our narrative shows.
"""

import logging
import os
import sys
import warnings


# ── ANSI colours (auto-disabled if stdout is not a TTY) ─────────────────────
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

class _C:
    if _USE_COLOR:
        RESET   = "\033[0m"
        BOLD    = "\033[1m"
        DIM     = "\033[2m"
        BLUE    = "\033[94m"   # TRE 1 / young
        AMBER   = "\033[93m"   # TRE 2 / old
        GREEN   = "\033[92m"   # TRE 3 / normal
        RED     = "\033[91m"   # central server / federated
        CYAN    = "\033[96m"
        GREY    = "\033[90m"
    else:
        RESET = BOLD = DIM = BLUE = AMBER = GREEN = RED = CYAN = GREY = ""


# Map cohort label → (TRE number, display name, colour)
TRE_INFO = {
    "USA_young":  (1, "Young cohort  (μ age 43)",  _C.BLUE),
    "USA_old":    (2, "Old cohort    (μ age 61)",  _C.AMBER),
    "USA_normal": (3, "Normal cohort (30–80 y)",   _C.GREEN),
}


def silence_third_party_logs():
    """
    Hide Flower's INFO/WARNING messages and Ray's startup chatter.
    Call this once at the start of main.py.
    """
    # Flower uses the standard logging module under the name "flwr"
    logging.getLogger("flwr").setLevel(logging.ERROR)

    # Ray writes to stdout/stderr directly; route via env vars
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    # Suppress Python warnings (deprecation notices etc.)
    warnings.filterwarnings("ignore")

    # General library noise
    for noisy in ("ray", "ray.tune", "absl", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


# ── Narrative output helpers ────────────────────────────────────────────────

def banner(title: str, subtitle: str = "") -> None:
    """Big top-of-script banner."""
    width = 70
    print()
    print(_C.BOLD + "═" * width + _C.RESET)
    print(_C.BOLD + f"  {title}" + _C.RESET)
    if subtitle:
        print(_C.DIM + f"  {subtitle}" + _C.RESET)
    print(_C.BOLD + "═" * width + _C.RESET)
    print()


def section(num: int, total: int, title: str) -> None:
    """Step header e.g. '[2/4] Doing the thing'"""
    print()
    print(_C.CYAN + _C.BOLD + f"[{num}/{total}] {title}" + _C.RESET)
    print(_C.DIM  + "─" * 70 + _C.RESET)


def round_header(round_num: int, total: int) -> None:
    print()
    print(_C.BOLD +
          f"  ┌─ ROUND {round_num}/{total} ─────────────────────────────────"
          + _C.RESET)


def round_footer() -> None:
    print(_C.BOLD +
          "  └────────────────────────────────────────────────"
          + _C.RESET)


def server_to_tre(tre_num: int, tre_name: str, colour: str) -> None:
    """'Server → TRE' message."""
    print(f"  {_C.RED}[Server]{_C.RESET} → {colour}{_C.BOLD}TRE {tre_num}{_C.RESET}  "
          f"{_C.DIM}sending current global model to {tre_name}{_C.RESET}")


def training_at_tre(
    tre_num   : int,
    tre_name  : str,
    colour    : str,
    n_samples : int,
    n_epochs  : int,
) -> None:
    """'Training at TRE' message."""
    print(f"  {colour}[TRE {tre_num}]{_C.RESET} {_C.BOLD}Training at TRE {tre_num}{_C.RESET}"
          f"  ({n_samples:,} samples · {n_epochs} local epochs)")


def tre_to_server(
    tre_num : int,
    colour  : str,
    auc     : float | None = None,
    f1      : float | None = None,
) -> None:
    """'TRE → Server' message with optional metrics."""
    msg = (f"  {colour}[TRE {tre_num}]{_C.RESET} → {_C.RED}{_C.BOLD}[Server]{_C.RESET}  "
           f"{_C.DIM}returning updated weights{_C.RESET}")
    if auc is not None and f1 is not None:
        msg += (f"  {_C.DIM}(local train AUC={auc:.3f}, F1={f1:.3f}){_C.RESET}")
    print(msg)


def aggregating(n_clients: int) -> None:
    print(f"  {_C.RED}[Server]{_C.RESET} {_C.BOLD}Aggregating {n_clients} models{_C.RESET}"
          f"  {_C.DIM}(FedAvg, weighted by sample count){_C.RESET}")


def round_eval(round_num: int, global_ll: float, per_tre: dict) -> None:
    """Print the post-round evaluation block."""
    print(f"  {_C.RED}[Server]{_C.RESET} {_C.BOLD}Evaluating new global model"
          f"{_C.RESET}  {_C.DIM}global log-loss = {global_ll:.4f}{_C.RESET}")
    for label, m in per_tre.items():
        if label not in TRE_INFO:
            continue
        tre_num, _, colour = TRE_INFO[label]
        print(f"           {colour}TRE {tre_num}{_C.RESET}: "
              f"AUC={m.get('test_auc', 0):.4f}  "
              f"F1={m.get('test_f1', 0):.4f}  "
              f"Acc={m.get('test_acc', 0):.4f}")


def cohort_summary_table(metas: list) -> None:
    print(f"  {_C.BOLD}{'TRE':<5}{'Cohort':<28}{'N':>9}{'<41y':>8}"
          f"{'>69y':>8}{'Cases':>8}{'Case%':>8}{_C.RESET}")
    print(_C.DIM + "  " + "─" * 70 + _C.RESET)
    for i, m in enumerate(metas, start=1):
        label = m["cohort_label"]
        if label in TRE_INFO:
            _, name, colour = TRE_INFO[label]
        else:
            name, colour = label, ""
        print(f"  {colour}{_C.BOLD}{i:<5}{_C.RESET}"
              f"{colour}{name:<28}{_C.RESET}"
              f"{m['n_samples']:>9,}"
              f"{m['pct_under41']:>7.1f}%"
              f"{m['pct_over69']:>7.1f}%"
              f"{m['n_cases']:>8,}"
              f"{m['case_rate']*100:>7.2f}%")
    print()


def final_metrics_table(metrics: dict) -> None:
    print()
    print(_C.BOLD + "  ┌─ FINAL CLASSIFICATION METRICS "
          "─────────────────────────────" + _C.RESET)
    print(f"  {_C.BOLD}{'Metric':<14}{'Value':>14}{_C.RESET}")
    print(_C.DIM + "  " + "─" * 32 + _C.RESET)
    for key, label in [
        ("accuracy",   "Accuracy"),
        ("auc",        "AUC"),
        ("log_loss",   "Log Loss"),
        ("precision",  "Precision"),
        ("recall",     "Recall"),
        ("f1",         "F1 Score"),
    ]:
        print(f"  {label:<14}{metrics.get(key, float('nan')):>14.4f}")
    print()


def coef_recovery_table(stats: dict) -> None:
    print(_C.BOLD + "  ┌─ COEFFICIENT RECOVERY (vs true PRS effect sizes) "
          "─────────" + _C.RESET)
    print(f"  {_C.BOLD}{'Metric':<14}{'Value':>14}{_C.RESET}")
    print(_C.DIM + "  " + "─" * 32 + _C.RESET)
    for key, label, fmt in [
        ("pearson_r", "Pearson r",  ".4f"),
        ("rmse",      "RMSE",       ".6f"),
        ("mape",      "MAPE %",     ".2f"),
        ("n",         "N variants", "d"),
    ]:
        v = stats.get(key, float("nan"))
        if fmt == "d":
            cell = f"{int(v):>14d}"
        else:
            cell = f"{v:>14{fmt}}"
        print(f"  {label:<14}{cell}")
    print()


def done(output_paths: list) -> None:
    print()
    print(_C.GREEN + _C.BOLD + "  ✓ Done!" + _C.RESET)
    print(_C.DIM + "  Outputs written to:" + _C.RESET)
    for p in output_paths:
        print(f"    {os.path.abspath(p)}")
    print()
