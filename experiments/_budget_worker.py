"""
Worker entry point for one strategy-budget evaluation cell.

Called by run_budget_sweep() via subprocess.run():
  python -m experiments._budget_worker <params_json_path>

The worker loads data, builds a fresh strategy, runs evaluate(), writes
params["result_path"], and exits so process memory is fully released.
"""

import json
import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m experiments._budget_worker <params_json_path>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        params = json.load(f)

    # Switch to the package root so relative imports and paths work normally.
    cwd = params["cwd"]
    os.chdir(cwd)
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    from src.models.llm_client import LLMClient
    from src.experiment_utils import build_strategies_from_config, load_experiment_config
    from src.evaluation.longmemeval_runner import LongMemEvalRunner
    from src.evaluation.locomo_runner import LoCoMoRunner

    cfg = load_experiment_config(params["config_path"])
    answer_llm = LLMClient(params["model_name"])
    judge_llm = LLMClient(params["judge_model"])

    all_strats = build_strategies_from_config(
        answer_llm,
        cfg,
        include_full_context=True,
        persist_root=params["persist_root"],
    )
    strat = next((s for s in all_strats if s.name == params["strategy_name"]), None)
    if strat is None:
        print(f"Strategy '{params['strategy_name']}' not found", file=sys.stderr)
        sys.exit(1)

    fresh = getattr(strat, "_factory", None)
    budget_strat = fresh() if callable(fresh) else strat
    for key, value in params.get("strategy_runtime", {}).items():
        if hasattr(budget_strat, key):
            setattr(budget_strat, key, value)

    benchmark = params.get("benchmark", "longmemeval")
    retain_context = benchmark == "longmemeval" and strat.name in {"dual_track", "bm25_chunk_trackb", "session_summary"}
    if benchmark == "locomo":
        runner = LoCoMoRunner(
            data_path=params["data_path"],
            judge_llm=judge_llm,
            answer_llm=answer_llm,
        )
        report = runner.evaluate(
            strategy=budget_strat,
            answer_llm=answer_llm,
            max_samples=params["max_samples"],
            budget_chars=params["budget"],
            seed=params["seed"],
            shuffle=params.get("shuffle", False),
        )
    else:
        runner = LongMemEvalRunner(
            data_path=params["data_path"],
            judge_llm=judge_llm,
            preload_max=None,
        )
        report = runner.evaluate(
            strategy=budget_strat,
            answer_llm=answer_llm,
            max_samples=params["max_samples"],
            budget_chars=params["budget"],
            seed=params["seed"],
            shuffle=params.get("shuffle", False),
            judge_mode=params["judge_mode"],
            sample_ids=params.get("sample_ids"),
            store_context=retain_context,
        )

    result = {
        "accuracy": report.accuracy,
        "avg_input_tokens": report.avg_input_tokens,
        "task_breakdown": report.task_breakdown,
        "sample_correctness": [
            {
                **({"memory_context": s.memory_context, "attribution": s.attribution,
                    "packing_stats": s.packing_stats} if retain_context else {}),
                "sample_id": s.sample_id,
                "task_type": s.task_type,
                "is_correct": s.is_correct,
                "prediction": s.prediction,
                "gold_answer": s.gold_answer,
                "judge_verdict_raw": s.judge_verdict_raw,
                "input_tokens": s.input_tokens,
                "memory_context_len": s.memory_context_len,
            }
            for s in report.samples
        ],
    }

    with open(params["result_path"], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
