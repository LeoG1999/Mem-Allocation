"""Table 16: 12k, 24k, and full-history settings."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from experiments.exp_paper_controls import (
    AllocationDualStrategy,
    append_jsonl,
    build_strategy,
    condition_spec,
    read_jsonl,
    select_samples,
    utc_now,
    write_json,
)
from src.evaluation.longmemeval_runner import LongMemEvalRunner
from src.memory.baselines.strategies import FullContextStrategy
from src.models.llm_client import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/longmemeval/longmemeval_s.json")
    parser.add_argument("--answer-model", default="deepseek-v4-flash")
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--budgets", nargs="+", type=int, default=[12000, 24000])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="data/results/extended_budget")
    parser.add_argument("--summary-cache-dir", default="data/cache/deepseek-v4-flash/summary_cache")
    parser.add_argument("--packing-mode", default="whole_unit_skip")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    runner = LongMemEvalRunner(args.data_path)
    samples = select_samples(runner, args.max_samples, [])
    write_json(output_dir / "manifest.json", {
        "paper_table": 15,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "budgets": args.budgets,
        "n_samples": len(samples),
        "split_track_b_top_k": None,
        "packing_mode": args.packing_mode,
        "created_at": utc_now(),
        "dry_run": args.dry_run,
    })
    if args.dry_run:
        return

    answer_llm = LLMClient(args.answer_model)
    judge_llm = LLMClient(args.judge_model)
    runner.judge_llm = judge_llm

    conditions = {
        "split_extended": lambda: AllocationDualStrategy(
            name="split_extended",
            llm_client=answer_llm,
            spec=condition_spec("split_extended"),
            summary_cache_dir=args.summary_cache_dir,
            use_llm_summary=True,
            packing_mode=args.packing_mode,
        ),
        "trackb1024": lambda: build_strategy(
            condition_spec("trackb1024"), answer_llm, args
        ),
        "full_history": FullContextStrategy,
    }

    for condition, factory in conditions.items():
        budgets = [args.budgets[0]] if condition == "full_history" else args.budgets
        for budget in budgets:
            output = output_dir / "raw" / f"{condition}_b{budget}.jsonl"
            done = {row["sample_id"] for row in read_jsonl(output)}
            strategy = factory()
            for sample in samples:
                if sample.sample_id in done:
                    continue
                report = runner.evaluate(
                    strategy=strategy,
                    answer_llm=answer_llm,
                    sample_ids=[sample.sample_id],
                    max_samples=1,
                    budget_chars=budget,
                    judge_mode="hybrid",
                    store_context=False,
                )
                append_jsonl(output, {
                    **asdict(report.samples[0]),
                    "condition": condition,
                    "nominal_budget_chars": budget,
                    "answer_model": answer_llm.model,
                    "judge_model": judge_llm.model,
                })


if __name__ == "__main__":
    main()
