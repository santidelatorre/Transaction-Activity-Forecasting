#!/usr/bin/env python
"""Subscription Foresight — jury demo CLI (agentic tool spine).

Screenshot-ready path (no 7-minute fit if cache exists):

  python scripts/demo_subscription_foresight.py metrics
  python scripts/demo_subscription_foresight.py clients
  python scripts/demo_subscription_foresight.py explain --client C000000
  python scripts/demo_subscription_foresight.py predict --client C000000

Warm cache once (fits PushEmbed on TRAIN, scores VALID):

  python scripts/demo_subscription_foresight.py warm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transaction_forecasting.ubs.demo_agent import DEFAULT_CACHE, ForesightAgent


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ulmans Subscription Foresight demo agent",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Directory for warm prediction cache",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("metrics", help="show_metrics tool — frozen scoreboard")
    p_clients = sub.add_parser("clients", help="list_clients tool")
    p_clients.add_argument("--pack", choices=("valid", "train"), default="valid")
    p_clients.add_argument("--limit", type=int, default=12)

    p_explain = sub.add_parser("explain", help="explain_streams tool")
    p_explain.add_argument("--client", required=True)
    p_explain.add_argument("--top-n", type=int, default=8)

    p_pred = sub.add_parser("predict", help="predict_client tool")
    p_pred.add_argument("--client", required=True)
    p_pred.add_argument(
        "--reveal-label",
        action="store_true",
        help="Display-only true label for internal QA (not used in model)",
    )
    p_pred.add_argument(
        "--fit-if-needed",
        action="store_true",
        help="Fit live model if cache missing (slow)",
    )

    p_warm = sub.add_parser("warm", help="Fit + cache VALID probs for instant demo")
    p_warm.add_argument(
        "--max-valid",
        type=int,
        default=None,
        help="Optional cap for smoke tests",
    )

    p_tour = sub.add_parser(
        "tour",
        help="One-shot demo tour: metrics → client → explain → predict",
    )
    p_tour.add_argument("--client", default=None)

    args = parser.parse_args(argv)
    agent = ForesightAgent(cache_dir=args.cache_dir)

    if args.cmd == "metrics":
        _print(agent.show_metrics())
        return 0
    if args.cmd == "clients":
        _print(agent.list_clients(pack=args.pack, limit=args.limit))
        return 0
    if args.cmd == "explain":
        _print(agent.explain_streams(args.client, top_n=args.top_n))
        return 0
    if args.cmd == "predict":
        _print(
            agent.predict_client(
                args.client,
                reveal_label=args.reveal_label,
                fit_if_needed=args.fit_if_needed,
            )
        )
        return 0
    if args.cmd == "warm":
        print("Fitting PushEmbed + caching VALID probabilities (several minutes)…", flush=True)
        _print(agent.warm(max_valid=args.max_valid))
        return 0
    if args.cmd == "tour":
        metrics = agent.show_metrics()
        clients = agent.list_clients(pack="valid", limit=5)
        client = args.client or clients["client_ids"][0]
        explain = agent.explain_streams(client)
        predict = agent.predict_client(client, fit_if_needed=False)
        _print(
            {
                "tour": "Subscription Foresight",
                "metrics": {
                    "macro_f1": metrics.get("push_embed_macro_f1"),
                    "accuracy": metrics.get("push_embed_accuracy"),
                    "honesty": metrics.get("honesty"),
                },
                "client": client,
                "explain": explain,
                "predict": predict,
                "next_step_if_no_cache": (
                    "python scripts/demo_subscription_foresight.py warm"
                    if predict.get("error")
                    else None
                ),
            }
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
