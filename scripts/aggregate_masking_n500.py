#!/usr/bin/env python3
"""Aggregate masking results for Table 13."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CONDITIONS = {
    "dual_track": "dual_track_split",
    "dual_track_mask_a": "chunks_only_split_budget",
    "dual_track_mask_b": "summaries_only_split_budget",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_type"]].append(bool(row["is_correct"]))
    correct = sum(bool(row["is_correct"]) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy_pct": round(100 * correct / len(rows), 3) if rows else None,
        "by_task": {
            task: {
                "n": len(values),
                "accuracy_pct": round(100 * sum(values) / len(values), 3),
            }
            for task, values in sorted(groups.items())
        },
        "sample_correctness": [
            {
                "sample_id": row["sample_id"],
                "task_type": row["task_type"],
                "is_correct": bool(row["is_correct"]),
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("data/results/masking"))
    parser.add_argument("--output", type=Path, default=Path("data/results/masking/table13.json"))
    parser.add_argument("--budget", type=int, default=4000)
    args = parser.parse_args()

    payload = {"budget": args.budget, "conditions": {}}
    cohort = None
    for source, paper_name in CONDITIONS.items():
        rows = read_jsonl(args.input_dir / f"per_sample_{source}_b{args.budget}.jsonl")
        ids = {row["sample_id"] for row in rows}
        if not ids or len(ids) != len(rows) or (cohort is not None and ids != cohort):
            raise ValueError(f"Unpaired or incomplete masking records for {source}")
        if not all(isinstance(row["is_correct"], bool) for row in rows):
            raise ValueError(f"Invalid masking verdict for {source}")
        cohort = ids
        payload["conditions"][paper_name] = summarize(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
