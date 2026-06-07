"""
tre_logger.py — Production version.

In production each TRE runs in its own process / container, so each
process only logs about itself. The server logs aggregation events;
each client logs its own training and evaluation events.
"""

import os
import sys
import logging
import warnings


_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class _C:
    if _USE_COLOR:
        RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
        BLUE = "\033[94m"; AMBER = "\033[93m"; GREEN = "\033[92m"
        RED = "\033[91m";  CYAN = "\033[96m"; GREY = "\033[90m"
    else:
        RESET = BOLD = DIM = BLUE = AMBER = GREEN = RED = CYAN = GREY = ""


# Colour each TRE consistently (so dashboards / logs are easy to scan)
TRE_COLOURS = {
    1: _C.BLUE,
    2: _C.AMBER,
    3: _C.GREEN,
}


def silence_third_party_logs():
    """Hide Flower's INFO messages and Python deprecation warnings."""
    logging.getLogger("flwr").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")


def banner(role: str, identifier: str, subtitle: str = "") -> None:
    """Big top-of-script banner identifying which role this process is."""
    width = 70
    print()
    print(_C.BOLD + "═" * width + _C.RESET)
    print(_C.BOLD + f"  {role}: {identifier}" + _C.RESET)
    if subtitle:
        print(_C.DIM + f"  {subtitle}" + _C.RESET)
    print(_C.BOLD + "═" * width + _C.RESET)
    print()


def info(msg: str) -> None:
    print(f"  {_C.DIM}{msg}{_C.RESET}")


def event(label: str, msg: str, colour: str = _C.CYAN) -> None:
    print(f"  {colour}{_C.BOLD}[{label}]{_C.RESET} {msg}")


# ── Server-side helpers ─────────────────────────────────────────────────────

def server_round_header(round_num: int, total_rounds: int) -> None:
    print()
    print(_C.RED + _C.BOLD +
          f"  ┌─ ROUND {round_num}/{total_rounds} ──────────────────────────"
          + _C.RESET)


def server_broadcasting(n_clients: int) -> None:
    print(f"  {_C.RED}[Server]{_C.RESET} "
          f"{_C.BOLD}Broadcasting global model to {n_clients} TREs{_C.RESET}")


def server_received_update(n_clients: int) -> None:
    print(f"  {_C.RED}[Server]{_C.RESET} "
          f"{_C.BOLD}Received updates from {n_clients} TREs{_C.RESET}  "
          f"{_C.DIM}(FedAvg aggregation){_C.RESET}")


def server_eval_summary(round_num: int, global_ll: float, per_tre: dict) -> None:
    print(f"  {_C.RED}[Server]{_C.RESET} "
          f"{_C.BOLD}Evaluation complete{_C.RESET}  "
          f"{_C.DIM}global log-loss = {global_ll:.4f}{_C.RESET}")
    for tre_num in sorted(per_tre.keys()):
        m = per_tre[tre_num]
        colour = TRE_COLOURS.get(tre_num, "")
        cohort = m.get("cohort", "?")
        print(f"           {colour}TRE {tre_num}{_C.RESET} "
              f"({_C.DIM}{cohort}{_C.RESET}): "
              f"AUC={m.get('test_auc', 0):.4f}  "
              f"F1={m.get('test_f1', 0):.4f}  "
              f"Acc={m.get('test_acc', 0):.4f}")


def server_round_footer() -> None:
    print(_C.RED + _C.BOLD +
          "  └──────────────────────────────────────────────"
          + _C.RESET)


# ── Client-side helpers ─────────────────────────────────────────────────────

def client_waiting(server: str) -> None:
    print(f"  {_C.DIM}Connecting to server at {server}…"
          f" (will wait for rounds to begin){_C.RESET}")


def client_round_received(tre_num: int, cohort_label: str) -> None:
    colour = TRE_COLOURS.get(tre_num, "")
    print()
    print(f"  {colour}[TRE {tre_num}]{_C.RESET} "
          f"{_C.BOLD}Received global model from server{_C.RESET}")


def client_training(tre_num: int, n_samples: int, n_epochs: int) -> None:
    colour = TRE_COLOURS.get(tre_num, "")
    print(f"  {colour}[TRE {tre_num}]{_C.RESET} "
          f"{_C.BOLD}Training locally{_C.RESET}  "
          f"{_C.DIM}({n_samples:,} samples · {n_epochs} epochs){_C.RESET}")


def client_returning(tre_num: int, auc: float, f1: float) -> None:
    colour = TRE_COLOURS.get(tre_num, "")
    print(f"  {colour}[TRE {tre_num}]{_C.RESET} "
          f"{_C.BOLD}Returning updated weights to server{_C.RESET}  "
          f"{_C.DIM}(local AUC={auc:.3f}, F1={f1:.3f}){_C.RESET}")


def client_evaluated(tre_num: int, auc: float, f1: float, acc: float) -> None:
    colour = TRE_COLOURS.get(tre_num, "")
    print(f"  {colour}[TRE {tre_num}]{_C.RESET} "
          f"{_C.DIM}Evaluated on local test set: "
          f"AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}{_C.RESET}")


def client_cohort_summary(tre_num: int, meta: dict) -> None:
    colour = TRE_COLOURS.get(tre_num, "")
    print()
    print(f"  {colour}{_C.BOLD}Cohort summary for TRE {tre_num}{_C.RESET}")
    print(f"    {_C.DIM}Label       :{_C.RESET} {meta['cohort_label']}")
    print(f"    {_C.DIM}N samples   :{_C.RESET} {meta['n_samples']:,}")
    print(f"    {_C.DIM}Features    :{_C.RESET} {meta['n_features']}")
    print(f"    {_C.DIM}Age (mean)  :{_C.RESET} {meta['age_mean']:.1f}")
    print(f"    {_C.DIM}<41 y       :{_C.RESET} {meta['pct_under41']:.1f}%")
    print(f"    {_C.DIM}>69 y       :{_C.RESET} {meta['pct_over69']:.1f}%")
    print(f"    {_C.DIM}Cases       :{_C.RESET} {meta['n_cases']:,} "
          f"({meta['case_rate']*100:.2f}%)")
    print(f"    {_C.DIM}Train / Test:{_C.RESET} "
          f"{meta['n_train']:,} / {meta['n_test']:,}")
    print()


def done(msg: str = "Done.") -> None:
    print()
    print(_C.GREEN + _C.BOLD + f"  ✓ {msg}" + _C.RESET)
    print()
