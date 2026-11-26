#!/usr/bin/env python3
"""Independent judge audit for the matched concentration control.

This script audits the paper's main crossover comparison:
dual_track vs. bm25_chunk_trackb (reported as bm25_trackb_full).

It reads the stored predictions from the main runs and sends the same outputs
to an independent judge. Strategy labels and memory contexts are omitted from
the judging prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.longmemeval_runner import (
    JUDGE_PROMPT,
    JUDGE_PROMPT_ABSTENTION,
    LongMemEvalRunner,
    parse_binary_verdict,
)
from src.models.llm_client import LLMClient


STRATEGIES = ["dual_track", "bm25_chunk_trackb"]
PAPER_NAMES = {
    "dual_track": "dual_track",
    "bm25_chunk_trackb": "bm25_trackb_full",
}


def stratified_sample_ids(data, n: int, seed: int) -> tuple[list[str], dict]:
    """Sample approximately proportionally by task type."""
    groups: Dict[str, list] = defaultdict(list)
    for sample in data:
        groups[sample.task_type].append(sample)

    total = sum(len(v) for v in groups.values())
    if not 0 < n <= total:
        raise ValueError(f"Requested {n} audit questions from a dataset of {total}")

    # Largest-remainder allocation keeps the sample close to the full task mix.
    alloc = {}
    remainders = []
    for task, rows in sorted(groups.items()):
        exact = n * len(rows) / total
        base = min(len(rows), int(exact))
        alloc[task] = base
        remainders.append((exact - base, task))
    remaining = n - sum(alloc.values())
    for _, task in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if alloc[task] < len(groups[task]):
            alloc[task] += 1
            remaining -= 1

    rng = random.Random(seed)
    sample_ids = []
    task_counts = {}
    for task, rows in sorted(groups.items()):
        chosen = rng.sample(rows, alloc[task])
        sample_ids.extend(str(s.sample_id) for s in chosen)
        task_counts[task] = len(chosen)
    rng.shuffle(sample_ids)
    return sample_ids, task_counts


def load_stored_outputs(
    budget: int,
    sample_ids: Iterable[str],
    main_path: Path,
    trackb_path: Path,
    questions: dict[str, str],
) -> dict:
    """Load predictions and main-judge labels for the same strategy outputs."""
    sample_ids = [str(s) for s in sample_ids]
    sample_id_set = set(sample_ids)
    if len(sample_id_set) != len(sample_ids):
        raise ValueError("Duplicate question IDs in the audit cohort")
    main_payload = json.loads(main_path.read_text())
    main = main_payload["raw_runs"][0] if "raw_runs" in main_payload and main_payload["raw_runs"] else main_payload
    trackb = json.loads(trackb_path.read_text())

    def rows_for(payload: dict, strategy: str) -> list[dict]:
        cell = payload[strategy]
        if cell["budgets"].count(budget) != 1:
            raise ValueError(f"Expected one saved cell for {strategy} at budget {budget}")
        rows = cell["sample_correctness"][cell["budgets"].index(budget)]
        by_id = {}
        for row in rows:
            sid = str(row["sample_id"])
            if sid not in questions:
                raise ValueError(f"Unknown question {sid} in {strategy} at budget {budget}")
            if sid not in sample_id_set:
                continue
            if sid in by_id:
                raise ValueError(f"Duplicate question {sid} in {strategy} at budget {budget}")
            by_id[sid] = dict(row, sample_id=sid, question=questions[sid],
                              main_judge_correct=bool(row["is_correct"]))
        missing = sample_id_set - by_id.keys()
        if missing:
            raise ValueError(f"{strategy} at budget {budget} is missing {len(missing)} questions: {min(missing)}")
        return [by_id[sid] for sid in sample_ids]

    return {
        "dual_track": rows_for(main, "dual_track"),
        "bm25_chunk_trackb": rows_for(trackb, "bm25_chunk_trackb"),
    }


def summarize_pair(results_by_strategy: dict, sample_ids: list[str]) -> dict:
    dt = {
        row["sample_id"]: bool(row["is_correct"])
        for row in results_by_strategy["dual_track"]["sample_correctness"]
    }
    tb = {
        row["sample_id"]: bool(row["is_correct"])
        for row in results_by_strategy["bm25_chunk_trackb"]["sample_correctness"]
    }
    shared = [sid for sid in sample_ids if sid in dt and sid in tb]
    wins = sum(1 for sid in shared if dt[sid] and not tb[sid])
    losses = sum(1 for sid in shared if (not dt[sid]) and tb[sid])
    ties = len(shared) - wins - losses
    dt_acc = sum(dt[sid] for sid in shared) / len(shared) if shared else 0.0
    tb_acc = sum(tb[sid] for sid in shared) / len(shared) if shared else 0.0
    return {
        "n_shared": len(shared),
        "dual_track_accuracy": dt_acc,
        "bm25_trackb_full_accuracy": tb_acc,
        "delta": dt_acc - tb_acc,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def cross_judge_agreement(independent: dict, sample_ids: list[str]) -> dict:
    pairs = []
    for strategy in STRATEGIES:
        row_map = {
            row["sample_id"]: row
            for row in independent[strategy]["sample_correctness"]
        }
        for sample_id in sample_ids:
            if sample_id in row_map:
                row = row_map[sample_id]
                pairs.append((bool(row["is_correct"]), bool(row["main_judge_correct"])))
    if not pairs:
        return {"n_decisions": 0, "agreement_pct": None, "cohen_kappa": None}
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_yes = sum(left for left, _ in pairs) / len(pairs)
    right_yes = sum(right for _, right in pairs) / len(pairs)
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "n_decisions": len(pairs),
        "agreement_pct": round(100 * observed, 3),
        "cohen_kappa": round(kappa, 4),
    }


def judge_prediction(
    judge_llm: LLMClient,
    prediction: str,
    gold: str,
    task_type: str,
    question: str,
) -> tuple[bool, str]:
    """Blind independent judging: no strategy labels are included."""
    if not prediction or prediction == "[API_ERROR]":
        return False, ""
    if task_type == "abstention":
        prompt = JUDGE_PROMPT_ABSTENTION.format(
            question=question,
            gold=gold,
            prediction=prediction,
        )
    else:
        prompt = JUDGE_PROMPT.format(gold=gold, prediction=prediction)
    resp = judge_llm.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    verdict = resp.content.strip()
    return parse_binary_verdict(verdict), verdict


def apply_independent_judge(payload: dict, judge_llm: LLMClient) -> dict:
    judged_samples = []
    for row in payload["samples"]:
        is_correct, verdict = judge_prediction(
            judge_llm=judge_llm,
            prediction=row["prediction"],
            gold=row["gold_answer"],
            task_type=row["task_type"],
            question=row["question"],
        )
        row = dict(row)
        row["is_correct"] = is_correct
        row["independent_judge_verdict"] = verdict
        judged_samples.append(row)

    sample_correctness = [
        {
            "sample_id": row["sample_id"],
            "task_type": row["task_type"],
            "is_correct": row["is_correct"],
            "main_judge_correct": row["main_judge_correct"],
        }
        for row in judged_samples
    ]
    acc = sum(row["is_correct"] for row in judged_samples) / len(judged_samples) if judged_samples else 0.0
    task_groups: Dict[str, list[bool]] = defaultdict(list)
    for row in judged_samples:
        task_groups[row["task_type"]].append(row["is_correct"])
    payload = dict(payload)
    payload["accuracy"] = acc
    payload["task_breakdown"] = {
        task: sum(vals) / len(vals)
        for task, vals in sorted(task_groups.items())
    }
    payload["sample_correctness"] = sample_correctness
    payload["samples"] = judged_samples
    payload["judging"] = {
        "judge_model": judge_llm.model,
        "blind": True,
        "labels_hidden": True,
        "judge_applied_after_generation": True,
    }
    return payload


def judge_cell(
    *,
    strategy_name: str,
    budget: int,
    stored_rows: list[dict],
    output_dir: Path,
    judge_model: str,
    force: bool,
    judge_llm: LLMClient,
) -> dict:
    partial_dir = output_dir / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    judge_safe = judge_model.replace("/", "_")
    partial_path = partial_dir / f"matched_judge_{strategy_name}_b{budget}_{judge_safe}.json"
    source = json.dumps(stored_rows, sort_keys=True, ensure_ascii=False).encode()
    source_sha256 = hashlib.sha256(source).hexdigest()

    def resume():
        cached = json.loads(partial_path.read_text())
        if cached.get("source_sha256") != source_sha256:
            raise ValueError(f"Incompatible audit checkpoint {partial_path}. Use a new output directory or --force.")
        print(f"  [{strategy_name} b={budget}] RESUME {partial_path}")
        return cached

    if partial_path.exists() and not force:
        return resume()
    payload = {
        "strategy": strategy_name,
        "paper_name": PAPER_NAMES[strategy_name],
        "budget": budget,
        "n_samples": len(stored_rows),
        "samples": stored_rows,
        "source_sha256": source_sha256,
        "stored_predictions_reused": True,
    }
    payload = apply_independent_judge(payload, judge_llm)
    with TemporaryDirectory(dir=partial_dir) as directory:
        pending = Path(directory) / "judgments.json"
        pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        if force:
            pending.replace(partial_path)
        else:
            try:
                os.link(pending, partial_path)
            except FileExistsError:
                return resume()
    print(f"  [{strategy_name} b={budget}] acc={payload['accuracy']:.3f} n={len(stored_rows)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/longmemeval/longmemeval_s.json")
    parser.add_argument("--output_dir", default="data/results/judge_validation")
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 4000, 6000])
    parser.add_argument("--n_per_budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--main_results",
        type=Path,
        default=ROOT / "data/results/main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json",
    )
    parser.add_argument(
        "--trackb_results",
        type=Path,
        default=ROOT / "data/results/trackb_deepseek/trackb_fullbudget_results.json",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_runner = LongMemEvalRunner(data_path=args.data_path, preload_max=None)
    data = data_runner._load_data()
    sample_ids, task_counts = stratified_sample_ids(data, args.n_per_budget, args.seed)
    questions = {str(sample.sample_id): sample.question for sample in data}
    if len(questions) != len(data):
        raise ValueError("Duplicate question IDs in the dataset")

    plan = {
        "data_path": args.data_path,
        "judge_model": args.judge_model,
        "budgets": args.budgets,
        "strategies": STRATEGIES,
        "n_per_budget": len(sample_ids),
        "sample_ids": sample_ids,
        "task_counts": task_counts,
        "blind_judging": "stored predictions are judged without strategy labels or memory contexts",
    }
    plan_path = output_dir / "audit_plan.json"
    if plan_path.exists() and not args.force and json.loads(plan_path.read_text()) != plan:
        raise ValueError(f"Incompatible audit plan {plan_path}. Use a new output directory or --force.")

    print("Matched-control judge audit plan")
    print(json.dumps({k: v for k, v in plan.items() if k != "sample_ids"}, indent=2))
    if args.dry_run:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"DRY RUN only. Wrote {output_dir / 'audit_plan.json'}")
        return

    stored_by_budget = {
        budget: load_stored_outputs(budget, sample_ids, args.main_results, args.trackb_results, questions)
        for budget in args.budgets
    }
    judge_llm = LLMClient(args.judge_model)

    by_budget = {}
    for budget in args.budgets:
        print(f"\n=== Budget {budget} ===")
        stored = stored_by_budget[budget]
        strategy_results = {}
        for strategy_name in STRATEGIES:
            strategy_results[strategy_name] = judge_cell(
                strategy_name=strategy_name,
                budget=budget,
                stored_rows=stored[strategy_name],
                output_dir=output_dir,
                judge_model=args.judge_model,
                force=args.force,
                judge_llm=judge_llm,
            )

        gpt_summary = summarize_pair(strategy_results, sample_ids)

        ds_as_results = {
            name: {
                "sample_correctness": [
                    {"sample_id": row["sample_id"], "is_correct": row["main_judge_correct"]}
                    for row in stored[name]
                ]
            }
            for name in STRATEGIES
        }
        deepseek_summary = summarize_pair(ds_as_results, sample_ids)
        agreement = cross_judge_agreement(strategy_results, sample_ids)
        by_budget[str(budget)] = {
            "independent_judge": gpt_summary,
            "independent_judge_model": args.judge_model,
            "deepseek_judge_existing": deepseek_summary,
            "direction_consistent": (gpt_summary["delta"] > 0) == (deepseek_summary["delta"] > 0),
            "cross_judge": agreement,
        }
        print(
            f"  {args.judge_model} judge delta={gpt_summary['delta']:+.3f} "
            f"W/L/T={gpt_summary['wins']}/{gpt_summary['losses']}/{gpt_summary['ties']} | "
            f"DeepSeek existing delta={deepseek_summary['delta']:+.3f}"
        )

    final = {
        "plan": plan,
        "summary_by_budget": by_budget,
        "partials_dir": str(output_dir / "_partial"),
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    out_path = output_dir / "matched_control_judge_validation.json"
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
