"""
Orthogonality / evidence-coverage analysis for dual_track (post-hoc).

Research question:
  For questions that dual_track answered correctly in the main n=500 run,
  where does the gold answer appear — only in Track A's session summaries,
  only in Track B's BM25 chunks, in both, or in neither (i.e. the model
  recovered the answer from prior reasoning)?

The analysis reads the per-track contexts attached to the main sweep's
correct predictions. Each evaluated item must include both stored track contexts.

Output schema (JSON):
  {
    "config": {...},
    "n_correct_total": int,
    "by_task_type": {
        "<task>": {
            "n": int,
            "a_only_pct": float,
            "b_only_pct": float,
            "both_pct": float,
            "neither_pct": float,
            "unknown_pct": float
        }, ...
    },
    "samples": [{sample_id, task_type, source, ...}, ...]
  }

Run:
  python -m experiments.exp_orthogonality \\
    --correctness_file data/results/longmemeval_main/budget_sweep_longmemeval_deepseek-v4-flash.json \\
    --strategy dual_track --budget 4000 \\
    --data_path data/longmemeval/longmemeval_s.json \\
    --model deepseek-v4-flash \\
    --output data/results/orthogonality/dual_track_b4000.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.longmemeval_runner import LongMemEvalRunner


STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "was", "are", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "with", "by", "from", "this", "that",
    "these", "those", "i", "he", "she", "it", "we", "they", "you", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "as",
}


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("’", "").replace("'", "")
    text = re.sub(r"[^\w]+|_", " ", text)
    return " ".join(text.split())


def _keywords(text: str, min_len: int = 3) -> List[str]:
    norm = _normalize(text)
    if norm in {"i dont know", "i do not know", "unknown", "not enough information", "insufficient information"}:
        return []
    return [w for w in norm.split() if w not in STOPWORDS and len(w) >= min_len]


def stored_track_contexts(row):
    attribution = row.get("attribution") or {}
    try:
        a, b = attribution["track_a_context"], attribution["track_b_context"]
    except KeyError as error:
        raise ValueError(f"Missing stored track contexts for {row['sample_id']}") from error
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError(f"Invalid stored track contexts for {row['sample_id']}")
    return a, b


def check_coverage(gold_answer: str, track_a_ctx: str, track_b_ctx: str) -> dict:
    """Decide whether gold answer is covered by Track A / Track B contexts.

    Coverage rule: at least ceil(K/2) of the gold answer's content keywords
    appear in the context (after normalization). For gold answers
    with no content keywords the coverage class is unknown.
    """
    gold_kw = _keywords(gold_answer)
    if not gold_kw:
        return {"a_covers": False, "b_covers": False, "n_keywords": 0,
                "source": "unknown", "a_hits": 0, "b_hits": 0}

    threshold = max(1, (len(gold_kw) + 1) // 2)  # ceil(K/2)

    a_norm = set(_normalize(track_a_ctx).split())
    b_norm = set(_normalize(track_b_ctx).split())

    a_hits = sum(1 for kw in gold_kw if kw in a_norm)
    b_hits = sum(1 for kw in gold_kw if kw in b_norm)

    a_covers = a_hits >= threshold
    b_covers = b_hits >= threshold

    if a_covers and b_covers:
        source = "both"
    elif a_covers:
        source = "A_only"
    elif b_covers:
        source = "B_only"
    else:
        source = "neither"

    return {
        "a_covers": a_covers, "b_covers": b_covers,
        "a_hits": a_hits, "b_hits": b_hits,
        "n_keywords": len(gold_kw),
        "source": source,
    }


@dataclass
class SampleRow:
    sample_id: str
    task_type: str
    source: str
    a_covers: bool
    b_covers: bool
    a_hits: int
    b_hits: int
    n_keywords: int


def load_correct_rows(
    correctness_file: str, strategy: str, budget: int,
) -> dict:
    """Load the stored rows that `strategy` answered correctly at `budget`."""
    with open(correctness_file) as f:
        payload = json.load(f)
    run = payload["raw_runs"][0] if "raw_runs" in payload and payload["raw_runs"] else payload
    if strategy not in run:
        raise ValueError(f"strategy '{strategy}' not in {correctness_file}")
    s = run[strategy]
    if budget not in s["budgets"]:
        raise ValueError(f"budget {budget} not in {s['budgets']}")
    b_idx = s["budgets"].index(budget)
    sc = s["sample_correctness"][b_idx]
    if not sc:
        raise ValueError(f"sample_correctness empty for {strategy} @ b={budget}")
    return {row["sample_id"]: row for row in sc if row["is_correct"]}


def run_orthogonality(
    correctness_file: str,
    strategy: str,
    budget: int,
    data_path: str,
    model: str,
    max_samples: Optional[int],
    verbose: bool,
) -> Dict:
    correct_rows = load_correct_rows(correctness_file, strategy, budget)
    if verbose:
        print(f"[load] {len(correct_rows)} samples were correct under {strategy} @ b={budget}")

    runner = LongMemEvalRunner(data_path)
    if runner._data is None:
        runner._data = runner._load_data()
    all_samples = runner._data
    target_samples = [s for s in all_samples if s.sample_id in correct_rows]
    if max_samples is not None:
        target_samples = target_samples[:max_samples]
    if verbose:
        print(f"[plan] analyzing {len(target_samples)} samples")

    rows: List[SampleRow] = []
    for i, sample in enumerate(target_samples):
        if verbose and (i % 25 == 0 or i == len(target_samples) - 1):
            print(f"  [{i+1}/{len(target_samples)}] {sample.sample_id} ({sample.task_type})")
        a, b = stored_track_contexts(correct_rows[sample.sample_id])
        cov = check_coverage(sample.gold_answer, a, b)
        rows.append(SampleRow(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            source=cov["source"],
            a_covers=cov["a_covers"],
            b_covers=cov["b_covers"],
            a_hits=cov["a_hits"],
            b_hits=cov["b_hits"],
            n_keywords=cov["n_keywords"],
        ))

    by_task: Dict[str, List[SampleRow]] = defaultdict(list)
    for r in rows:
        by_task[r.task_type].append(r)

    by_task_type = {}
    for task, rs in by_task.items():
        n = len(rs)
        bucket_counts = {"A_only": 0, "B_only": 0, "both": 0, "neither": 0, "unknown": 0}
        for r in rs:
            bucket_counts[r.source] = bucket_counts.get(r.source, 0) + 1
        by_task_type[task] = {
            "n": n,
            "a_only_pct": bucket_counts["A_only"] / n,
            "b_only_pct": bucket_counts["B_only"] / n,
            "both_pct": bucket_counts["both"] / n,
            "neither_pct": bucket_counts["neither"] / n,
            "unknown_pct": bucket_counts["unknown"] / n,
            "raw_counts": bucket_counts,
        }

    overall_n = len(rows)
    overall_counts = {"A_only": 0, "B_only": 0, "both": 0, "neither": 0, "unknown": 0}
    for r in rows:
        overall_counts[r.source] = overall_counts.get(r.source, 0) + 1

    return {
        "config": {
            "correctness_file": correctness_file,
            "strategy": strategy,
            "budget": budget,
            "data_path": data_path,
            "model": model,
            "n_samples_analyzed": overall_n,
            "max_samples_cap": max_samples,
        },
        "overall": {
            "n": overall_n,
            **{k + "_pct": (v / overall_n if overall_n else 0.0)
               for k, v in overall_counts.items()},
            "raw_counts": overall_counts,
        },
        "by_task_type": by_task_type,
        "samples": [asdict(r) for r in rows],
    }


def print_report(report: Dict) -> None:
    o = report["overall"]
    print(f"\nOverall (n={o['n']}):")
    print(f"  A_only={o['A_only_pct']:.1%}  B_only={o['B_only_pct']:.1%}  "
          f"both={o['both_pct']:.1%}  neither={o['neither_pct']:.1%}  unknown={o['unknown_pct']:.1%}")
    print()
    print(f"{'Task type':<26} {'N':>4} {'A_only':>8} {'B_only':>8} {'both':>8} {'neither':>8}")
    print("-" * 70)
    for task, d in sorted(report["by_task_type"].items()):
        print(f"{task:<26} {d['n']:>4}  {d['a_only_pct']:>7.1%} "
              f"{d['b_only_pct']:>7.1%} {d['both_pct']:>7.1%} {d['neither_pct']:>7.1%}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc orthogonality / evidence-coverage analysis")
    parser.add_argument("--correctness_file", required=True,
                        help="Path to merged_results.json (or compatible) with per-sample is_correct")
    parser.add_argument("--strategy", default="dual_track",
                        help="Which strategy's correct subset to analyze (default: dual_track)")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--data_path", default="data/longmemeval/longmemeval_s.json")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Model label for the stored main-run predictions")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional limit; default uses all correct samples")
    parser.add_argument("--output", default="data/results/orthogonality/orthogonality.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = run_orthogonality(
        correctness_file=args.correctness_file,
        strategy=args.strategy,
        budget=args.budget,
        data_path=args.data_path,
        model=args.model,
        max_samples=args.max_samples,
        verbose=not args.quiet,
    )
    print_report(report)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
