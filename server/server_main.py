"""
server_main.py — Central Flower aggregation server.

Listens on a TCP port, waits for the configured number of TRE clients to
connect, then orchestrates federated training rounds via FedAvg.

Usage:
    python server_main.py \\
        --host        0.0.0.0 \\
        --port        8080 \\
        --rounds      10 \\
        --num-clients 3 \\
        --n-features  313
        [--certs-dir  /certs]
"""

import argparse
import os
from pathlib import Path

import flwr as fl

from server.strategy   import TREStrategy
from server.tre_logger import silence_third_party_logs, banner, info, done


def parse_args():
    p = argparse.ArgumentParser(description="PRS federation central server")
    p.add_argument("--host",         default=os.environ.get("FLWR_HOST", "0.0.0.0"))
    p.add_argument("--port",         type=int,
                                     default=int(os.environ.get("FLWR_PORT", "8080")))
    p.add_argument("--rounds",       type=int,
                                     default=int(os.environ.get("ROUNDS", "10")))
    p.add_argument("--num-clients",  type=int,
                                     default=int(os.environ.get("NUM_CLIENTS", "3")))
    p.add_argument("--n-features",   type=int,
                                     default=int(os.environ.get("N_FEATURES", "313")),
                                     help="Must match what clients send back "
                                          "(313 SNPs for this demo).")
    p.add_argument("--results-dir",  default=os.environ.get("RESULTS_DIR", "/results"))
    p.add_argument("--certs-dir",    default=os.environ.get("CERTS_DIR", ""),
                   help="Directory with server.crt/server.key/ca.crt for TLS")
    return p.parse_args()


def _load_certs(certs_dir: str):
    """
    Load (CA cert, server cert, server key) for Flower's TLS.
    Returns None if certs aren't present (warns the user).
    """
    if not certs_dir:
        return None
    d = Path(certs_dir)
    ca  = d / "ca.crt"
    crt = d / "server.crt"
    key = d / "server.key"
    if not (ca.exists() and crt.exists() and key.exists()):
        print(f"  [WARN] --certs-dir set but cert files missing in {d}; "
              "starting server without TLS")
        return None
    return (ca.read_bytes(), crt.read_bytes(), key.read_bytes())


def main():
    args = parse_args()
    silence_third_party_logs()

    banner(
        role       = "Central server",
        identifier = f"{args.host}:{args.port}",
        subtitle   = f"Waiting for {args.num_clients} TREs · {args.rounds} rounds · FedAvg",
    )

    certs = _load_certs(args.certs_dir)
    if certs:
        info("TLS enabled (server.crt + server.key loaded)")
    else:
        info("TLS NOT enabled — server is in plaintext mode "
             "(fine for a private Docker network; not for production!)")

    strategy = TREStrategy(
        n_features            = args.n_features,
        n_rounds              = args.rounds,
        results_dir           = args.results_dir,
        fraction_fit          = 1.0,
        fraction_evaluate     = 1.0,
        min_fit_clients       = args.num_clients,
        min_evaluate_clients  = args.num_clients,
        min_available_clients = args.num_clients,
    )

    server_address = f"{args.host}:{args.port}"
    info(f"Server starting on {server_address} …")

    fl.server.start_server(
        server_address    = server_address,
        config            = fl.server.ServerConfig(num_rounds=args.rounds),
        strategy          = strategy,
        certificates      = certs,
    )

    # Save final artifacts after the run completes
    strategy.save_final_artifacts(n_features=args.n_features)

    done("Federation complete.")


if __name__ == "__main__":
    main()
