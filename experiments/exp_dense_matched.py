"""Table 3: replace the rankers for both memory views together."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from experiments.exp_paper_controls import (
    append_jsonl,
    read_jsonl,
    select_samples,
    sha256_text,
    utc_now,
    write_json,
)
from src.evaluation.longmemeval_runner import LongMemEvalRunner
from src.memory.dense_matched import DenseChunkTrackBStrategy, DenseDualStrategy
from src.models.llm_client import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/longmemeval/longmemeval_s.json")
    parser.add_argument("--answer-model", default="deepseek-v4-flash")
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--dense-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1000, 2000, 4000, 6000])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="data/results/dense_matched")
    parser.add_argument("--summary-cache-dir", default="data/cache/deepseek-v4-flash/summary_cache")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    runner = LongMemEvalRunner(args.data_path)
    samples = select_samples(runner, args.max_samples, [])
    write_json(output_dir / "manifest.json", {
        "paper_table": 3,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "dense_model": args.dense_model,
        "budgets": args.budgets,
        "n_samples": len(samples),
        "packing_mode": "whole_unit_skip",
        "created_at": utc_now(),
        "dry_run": args.dry_run,
    })
    if args.dry_run:
        return

    answer_llm = LLMClient(args.answer_model)
    judge_llm = LLMClient(args.judge_model)
    runner.judge_llm = judge_llm

    for condition in ("dense_split", "dense_trackb"):
        for budget in args.budgets:
            output = output_dir / "raw" / f"{condition}_b{budget}.jsonl"
            done = {row["sample_id"] for row in read_jsonl(output)}
            if condition == "dense_split":
                strategy = DenseDualStrategy(
                    llm_client=answer_llm,
                    model_name=args.dense_model,
                    summary_cache_dir=args.summary_cache_dir,
                    packing_mode="whole_unit_skip",
                )
            else:
                strategy = DenseChunkTrackBStrategy(
                    model_name=args.dense_model,
                    chunk_size_chars=1024,
                    chunk_overlap_chars=128,
                    packing_mode="whole_unit_skip",
                )
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
                row = asdict(report.samples[0])
                append_jsonl(output, {
                    **row,
                    "condition": condition,
                    "nominal_budget_chars": budget,
                    "prediction_sha256": sha256_text(row.get("prediction", "")),
                    "answer_model": answer_llm.model,
                    "judge_model": judge_llm.model,
                })


if __name__ == "__main__":
    main()
