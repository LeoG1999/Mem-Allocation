#!/usr/bin/env python3
"""Compute paired uncertainty for the paper's crossover comparisons.

The script uses existing per-item correctness artifacts; it does not call any
model.  It reports paired bootstrap confidence intervals over item-level
accuracy differences and exact McNemar p-values for paired binary outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAIN = ROOT / "data/results/main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json"
DEFAULT_TRACKB = ROOT / "data/results/trackb_full_control/trackb_fullbudget_results.json"
DEFAULT_BM25_CHUNK = ROOT / "data/results/main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json"
DEFAULT_OUT = ROOT / "data/results/statistical_tests/paired_crossover_stats.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def unwrap_run(payload: dict, strategy: str) -> dict:
    if strategy in payload:
        return payload
    if "raw_runs" in payload and payload["raw_runs"]:
        return payload["raw_runs"][0]
    raise KeyError(f"Could not find strategy {strategy!r}")


def sample_map(payload: dict, strategy: str, budget: int) -> Dict[str, bool]:
    run = unwrap_run(payload, strategy)
    result = run[strategy]
    b_idx = result["budgets"].index(budget)
    scores = {}
    for row in result["sample_correctness"][b_idx]:
        sid, verdict = str(row["sample_id"]), row["is_correct"]
        if sid in scores or not isinstance(verdict, bool):
            raise ValueError(f"Duplicate ID or invalid decision for {strategy}/{budget}/{sid}")
        scores[sid] = verdict
    return scores


def exact_mcnemar_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2 * prob)


def paired_bootstrap_ci(
    diffs: List[int],
    *,
    reps: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    draws = []
    for _ in range(reps):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo = draws[int((alpha / 2) * reps)]
    hi = draws[int((1 - alpha / 2) * reps)]
    return lo, hi


def compare(
    main: dict,
    comparator: dict,
    comparator_strategy: str,
    budgets: Iterable[int],
    *,
    reps: int,
    seed: int,
) -> list[dict]:
    rows = []
    for budget in budgets:
        dt = sample_map(main, "dual_track", budget)
        cp = sample_map(comparator, comparator_strategy, budget)
        if not dt or dt.keys() != cp.keys():
            raise ValueError(f"Paired policies must have identical nonempty item IDs at {budget}")
        sample_ids = sorted(dt)
        diffs = [int(dt[sid]) - int(cp[sid]) for sid in sample_ids]
        wins = sum(1 for sid in sample_ids if dt[sid] and not cp[sid])
        losses = sum(1 for sid in sample_ids if (not dt[sid]) and cp[sid])
        ties = len(sample_ids) - wins - losses
        lo, hi = paired_bootstrap_ci(diffs, reps=reps, seed=seed + budget)
        rows.append(
            {
                "budget": budget,
                "n": len(sample_ids),
                "dual_track_accuracy": sum(dt[sid] for sid in sample_ids) / len(sample_ids),
                "comparator_accuracy": sum(cp[sid] for sid in sample_ids) / len(sample_ids),
                "delta": sum(diffs) / len(sample_ids),
                "ci95_low": lo,
                "ci95_high": hi,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "mcnemar_p": exact_mcnemar_p(wins, losses),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--trackb", type=Path, default=DEFAULT_TRACKB)
    parser.add_argument("--bm25_chunk", type=Path, default=DEFAULT_BM25_CHUNK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 2000, 4000, 6000])
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main_results = load_json(args.main)
    trackb_results = load_json(args.trackb)
    bm25_chunk_results = load_json(args.bm25_chunk)

    payload = {
        "source": {
            "main": display_path(args.main),
            "trackb": display_path(args.trackb),
            "bm25_chunk": display_path(args.bm25_chunk),
        },
        "method": {
            "confidence_interval": "paired bootstrap over shared item ids",
            "bootstrap_reps": args.bootstrap_reps,
            "alpha": 0.05,
            "paired_test": "two-sided exact McNemar test",
        },
        "comparisons": {
            "dual_track_vs_bm25_trackb_full": compare(
                main_results,
                trackb_results,
                "bm25_chunk_trackb",
                args.budgets,
                reps=args.bootstrap_reps,
                seed=args.seed,
            ),
            "dual_track_vs_bm25_chunk_64w_top8": compare(
                main_results,
                bm25_chunk_results,
                "bm25_chunk",
                args.budgets,
                reps=args.bootstrap_reps,
                seed=args.seed + 100,
            ),
            "dual_track_vs_embedding_chunk": compare(
                main_results,
                main_results,
                "embedding_chunk",
                args.budgets,
                reps=args.bootstrap_reps,
                seed=args.seed + 200,
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for name, rows in payload["comparisons"].items():
        print(f"\n{name}")
        for row in rows:
            print(
                "  "
                f"B={row['budget']:4d} n={row['n']:3d} "
                f"delta={row['delta'] * 100:+.1f}pp "
                f"CI=[{row['ci95_low'] * 100:+.1f}, {row['ci95_high'] * 100:+.1f}] "
                f"win/loss/tie={row['wins']}/{row['losses']}/{row['ties']} "
                f"p={row['mcnemar_p']:.4g}"
            )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
