"""
Full-Budget Track B Allocation Control experiment.

Runs bm25_chunk_trackb on LongMemEval-S n=500 at budgets
{1000, 2000, 4000, 6000}.

The control uses Track B's chunk size, overlap, BM25 scoring, and chronological
serialization. It receives the full query-time memory budget and scans the
same ranked candidates as the split policy.

This is a pure BM25 strategy — no API calls for summarization,
only answer + judge calls via the configured API models.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_utils import build_strategies_from_config, load_experiment_config
from experiments.exp_budget_sweep import run_budget_sweep
from src.models.llm_client import LLMClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--judge_model",
        default=None,
        help="Judge model. Defaults to --model to preserve the original behavior.",
    )
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1000, 2000, 4000, 6000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", default="data/longmemeval/longmemeval_s.json")
    parser.add_argument("--output_dir", default="data/results/trackb_full_control")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--packing_mode", choices=["whole_unit_skip", "break_on_first_oversize", "truncate_last"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    judge_model = args.judge_model or args.model
    answer_llm = LLMClient(model=args.model)
    judge_llm = LLMClient(model=judge_model)

    strategies = build_strategies_from_config(
        answer_llm, load_experiment_config(args.config),
        persist_root=f"data/cache/{args.model.replace('/', '_')}",
    )
    strategy = next(s for s in strategies if s.name == "bm25_chunk_trackb")
    if args.packing_mode is not None:
        strategy.packing_mode = args.packing_mode

    print(f"Running bm25_chunk_trackb allocation control")
    print(f"  answer model: {args.model}")
    print(f"  judge model: {judge_model}")
    print(f"  samples: {args.max_samples}")
    print(f"  budgets: {args.budgets}")
    print(f"  output: {args.output_dir}")
    print()

    results = run_budget_sweep(
        strategies=[strategy],
        answer_llm=answer_llm,
        judge_llm=judge_llm,
        data_path=args.data_path,
        budgets=args.budgets,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        shuffle=True,
        judge_mode="hybrid",
        model_name=args.model,
        judge_model_name=judge_model,
        config_path=args.config,
        benchmark="longmemeval",
    )

    # Save final results
    out_file = os.path.join(args.output_dir, "trackb_fullbudget_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print summary
    r = results.get("bm25_chunk_trackb", {})
    print("\n" + "=" * 50)
    print("bm25_chunk_trackb (Full-Budget Track B Control)")
    print("=" * 50)
    for b, acc in zip(r.get("budgets", []), r.get("accuracies", [])):
        print(f"  b={b:>5}: {acc:.3f}")


if __name__ == "__main__":
    main()
