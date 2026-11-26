"""
Budget sweep experiment for the paper-facing accuracy curves.

Research question:
  How do memory strategies compare as the query-time memory budget changes?

Outputs:
  1. Accuracy curves across budgets.
  2. Per-task breakdowns.
  3. Per-sample correctness used by paired statistical tests.

Full run:
  python -m experiments.exp_budget_sweep \
    --model deepseek-v4-flash \
    --judge_model deepseek-v4-flash \
    --data_path data/longmemeval/longmemeval_s.json \
    --max_samples 500 \
    --seeds 42 --shuffle \
    --budgets 1000 2000 4000 6000

"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import gc
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.check_data import sha256 as file_sha256
from src.evaluation.stats import bootstrap_ci, mcnemar_exact_test, summarize_scalar
from src.experiment_utils import (
    build_strategies_from_config,
    get_strategy_group,
    load_experiment_config,
    save_json,
)
from src.models.llm_client import LLMClient
from src.memory.baselines.strategies import BaseMemoryStrategy
from src.evaluation.longmemeval_runner import LongMemEvalRunner


# Main experiment logic.

def select_strategies(strategies, names):
    aliases = {"bm25_chunk_64w_top8": "bm25_chunk", "bm25_trackb_full": "bm25_chunk_trackb"}
    requested = {aliases.get(name, name) for name in names}
    unknown = requested - {strategy.name for strategy in strategies}
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(sorted(unknown))}")
    return [strategy for strategy in strategies if strategy.name in requested]


def _load_checkpoint(path, settings):
    with open(path) as handle:
        result = json.load(handle)
    if result.get("run_settings") != settings:
        raise ValueError(f"Incompatible checkpoint {path}. Use a new output directory.")
    return result


def run_budget_sweep(
    strategies: List[BaseMemoryStrategy],
    answer_llm: LLMClient,
    judge_llm: Optional[LLMClient],
    data_path: str,
    budgets: List[int],
    max_samples: int,
    output_dir: str,
    seed: int = 0,
    shuffle: bool = False,
    judge_mode: str = "hybrid",
    progress_callback=None,
    # Parameters used by the isolated worker subprocess.
    model_name: str = "deepseek-v4-flash",
    judge_model_name: str = "deepseek-v4-flash",
    config_path: str = "configs/experiment_config.yaml",
    benchmark: str = "longmemeval",
    sample_ids: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Run each strategy at each memory budget and return the result matrix.

    Output structure:
    {
      strategy_name: {
        "budgets": [500, 1000, ...],
        "accuracies": [0.41, 0.55, ...],
        "avg_input_tokens": [840, 910, ...],
        "sample_correctness": [[...], ...],
      }
    }
    """
    results = {}
    persist_root = f"data/cache/{model_name.replace('/', '_')}"
    config_json = json.dumps(load_experiment_config(config_path), sort_keys=True)
    run_settings = {
        "context_protocol_version": 7,
        "data_sha256": file_sha256(Path(data_path)),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "benchmark": benchmark,
        "answer_model": model_name,
        "answer_deployment": getattr(answer_llm, "api_model", model_name),
        "judge_model": judge_model_name,
        "judge_deployment": getattr(judge_llm, "api_model", judge_model_name),
        "judge_mode": judge_mode,
        "max_samples": max_samples,
        "sample_ids": sorted({str(sid) for sid in sample_ids or []}),
        "seed": seed,
        "shuffle": shuffle,
    }

    for strat in strategies:
        print(f"\n{'='*55}")
        print(f"Strategy: {strat.name}")
        print(f"{'='*55}")

        accs, input_toks = [], []
        task_breakdowns, sample_correctness = [], []

        # Per-budget checkpoints for resumable long-running API experiments.
        model_safe = model_name.replace("/", "_")
        partial_dir = os.path.join(output_dir, "_partial")
        os.makedirs(partial_dir, exist_ok=True)
        runtime = {
            key: getattr(strat, key)
            for key in ("packing_mode", "budget_includes_wrapper")
            if hasattr(strat, key)
        }

        for budget in budgets:
            # Reuse a completed cell if it already exists.
            partial_file = os.path.join(
                partial_dir,
                f"partial_{benchmark}_{model_safe}_seed{seed}_{strat.name}_b{budget}.json",
            )
            settings = dict(run_settings, strategy=strat.name, budget=budget, strategy_runtime=runtime)
            if os.path.exists(partial_file):
                r = _load_checkpoint(partial_file, settings)
                accs.append(r["accuracy"])
                input_toks.append(r["avg_input_tokens"])
                task_breakdowns.append(r["task_breakdown"])
                sample_correctness.append(r["sample_correctness"])
                print(f"  budget={budget:>5} chars ... [RESUME] acc={r['accuracy']:.3f}")
                continue

            print(f"  budget={budget:>5} chars ... ", end="", flush=True)

            # Run each cell in a subprocess so memory is released after exit.
            with tempfile.NamedTemporaryFile(suffix="_params.json", delete=False) as handle:
                params_file = handle.name
            with tempfile.NamedTemporaryFile(dir=partial_dir, suffix="_result.json", delete=False) as handle:
                result_file = handle.name
            try:
                params = {
                    "cwd": os.getcwd(),
                    "model_name": model_name,
                    "judge_model": judge_model_name,
                    "data_path": data_path,
                    "budget": budget,
                    "max_samples": max_samples,
                    "seed": seed,
                    "shuffle": shuffle,
                    "judge_mode": judge_mode,
                    "strategy_name": strat.name,
                    "config_path": config_path,
                    "persist_root": persist_root,
                    "result_path": result_file,
                    "benchmark": benchmark,
                    "sample_ids": sample_ids,
                    "strategy_runtime": runtime,
                }
                with open(params_file, "w") as f:
                    json.dump(params, f)

                proc = subprocess.run(
                    [sys.executable, "-m", "experiments._budget_worker", params_file],
                    timeout=36000,  # 10 hour timeout
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"Worker exited with code {proc.returncode}")

                with open(result_file) as f:
                    r = json.load(f)
                r["run_settings"] = settings
                with open(result_file, "w") as f:
                    json.dump(r, f)
                try:
                    os.link(result_file, partial_file)
                except FileExistsError:
                    r = _load_checkpoint(partial_file, settings)
            finally:
                for p in (params_file, result_file):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

            accs.append(r["accuracy"])
            input_toks.append(r["avg_input_tokens"])
            task_breakdowns.append(r["task_breakdown"])
            sample_correctness.append(r["sample_correctness"])
            print(f"acc={r['accuracy']:.3f}  input_tokens={r['avg_input_tokens']:.0f}")

        results[strat.name] = {
            "budgets": budgets,
            "accuracies": accs,
            "avg_input_tokens": input_toks,
            "task_breakdowns": task_breakdowns,
            "sample_correctness": sample_correctness,
        }

        if progress_callback is not None:
            progress_callback(
                strategy_name=strat.name,
                result=results[strat.name],
                completed_budgets=list(budgets),
            )

    return results


def aggregate_runs(run_results: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    if not run_results:
        return {}

    strategy_names = list(run_results[0].keys())
    aggregated: Dict[str, Dict] = {}
    for name in strategy_names:
        budgets = run_results[0][name]["budgets"]
        n_points = len(budgets)
        agg_accs = []
        agg_inputs = []
        for i in range(n_points):
            acc_vals = [r[name]["accuracies"][i] for r in run_results]
            input_vals = [r[name]["avg_input_tokens"][i] for r in run_results]
            agg_accs.append({
                "budget": budgets[i],
                **summarize_scalar(acc_vals),
                "ci": bootstrap_ci(acc_vals, seed=i),
            })
            agg_inputs.append({
                "budget": budgets[i],
                **summarize_scalar(input_vals),
            })

        aggregated[name] = {
            "budgets": budgets,
            "accuracy_curve": agg_accs,
            "input_curve": agg_inputs,
        }

    if "dual_track" in strategy_names:
        sig = {}
        competitors = [n for n in strategy_names if n not in {"dual_track", "full_context"}]
        for comp in competitors:
            by_budget = []
            for i, budget in enumerate(run_results[0]["dual_track"]["budgets"]):
                pvals = []
                for run in run_results:
                    dt_rows = run["dual_track"]["sample_correctness"][i]
                    cp_rows = run[comp]["sample_correctness"][i]
                    dt_map = {r["sample_id"]: r["is_correct"] for r in dt_rows}
                    cp_map = {r["sample_id"]: r["is_correct"] for r in cp_rows}
                    common_ids = sorted(set(dt_map) & set(cp_map))
                    pvals.append(
                        mcnemar_exact_test(
                            [dt_map[sid] for sid in common_ids],
                            [cp_map[sid] for sid in common_ids],
                        )
                    )
                by_budget.append({
                    "budget": budget,
                    **summarize_scalar(pvals),
                })
            sig[comp] = by_budget
        aggregated["dual_track_significance"] = sig
    return aggregated

def save_results(results: Dict, output_dir: str, model: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"budget_sweep_{model.replace('/', '_')}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


def save_progress_snapshot(
    output_dir: str,
    model_name: str,
    seed: int,
    strategy_name: str,
    result: Dict,
    completed_budgets: List[int],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
        output_dir,
        f"budget_sweep_progress_{model_name.replace('/', '_')}_seed{seed}_{strategy_name}.json",
    )
    payload = {
        "model_name": model_name,
        "seed": seed,
        "strategy_name": strategy_name,
        "completed_budgets": completed_budgets,
        "result": result,
    }
    save_json(path, payload)
    return path


def save_run_manifest(
    output_dir: str,
    model_name: str,
    data_path: str,
    budgets: List[int],
    max_samples: int,
    seeds: List[int],
    strategies: List[str],
    completed: List[Dict[str, object]],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"budget_sweep_manifest_{model_name.replace('/', '_')}.json")
    payload = {
        "model_name": model_name,
        "data_path": data_path,
        "budgets": budgets,
        "max_samples": max_samples,
        "seeds": seeds,
        "strategies": strategies,
        "completed": completed,
    }
    save_json(path, payload)
    return path


def main():
    parser = argparse.ArgumentParser(description="Budget sweep experiment")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--judge_model", default="deepseek-v4-flash")
    parser.add_argument("--data_path", help="Defaults to the benchmark path in the configuration")
    parser.add_argument(
        "--budgets", nargs="+", type=int,
        help="Memory budgets in characters; defaults to the configured benchmark grid",
    )
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None,
                        help="Shuffle data before sampling (use with --seeds for reproducibility)")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--judge_mode", default="hybrid", choices=["hybrid", "exact", "string"])
    parser.add_argument("--output_dir", default="data/results")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Run only selected strategies. Defaults to config-defined strategies.")
    parser.add_argument("--benchmark", default="longmemeval",
                        choices=["longmemeval", "locomo"],
                        help="Evaluation benchmark. Default: longmemeval")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    locomo = args.benchmark == "locomo"
    evaluation = cfg.get("evaluation", {})
    if args.data_path is None:
        data = cfg.get("data", {})
        args.data_path = (data.get("locomo", "data/locomo/locomo_test.json") if locomo
                          else data.get("longmemeval_s", "data/longmemeval/longmemeval_s.json"))
    if args.budgets is None:
        args.budgets = (evaluation.get("locomo_budgets", [500, 1000, 2000, 4000]) if locomo
                        else evaluation.get("budgets", [1000, 2000, 4000, 6000]))
    if args.max_samples is None:
        sampling = cfg.get("sampling", {})
        args.max_samples = (sampling.get("locomo_max_samples", 200) if locomo
                            else sampling.get("max_samples", 500))
    if args.seeds is None:
        args.seeds = evaluation.get("seeds", [42])
    if args.shuffle is None:
        args.shuffle = evaluation.get("shuffle", True)
    models = args.models or [args.model]
    all_outputs = {}

    for model_name in models:
        answer_llm = LLMClient(model_name)
        judge_llm = LLMClient(args.judge_model)
        all_strategies = build_strategies_from_config(
            answer_llm,
            cfg,
            include_full_context=True,
            persist_root=f"data/cache/{model_name.replace('/', '_')}",
        )

        selected_names = args.strategies or get_strategy_group(
            cfg,
            "locomo_matched_strategies" if locomo else "main_table_strategies",
            fallback=["bm25_chunk_trackb", "dual_track"],
        )
        if selected_names:
            all_strategies = select_strategies(all_strategies, selected_names)

        print(f"Model: {model_name}")
        print(f"Budgets: {args.budgets}")
        print(f"Strategies: {[s.name for s in all_strategies]}")
        print(f"Samples: {args.max_samples}")
        print(f"Seeds: {args.seeds}")
        completed_manifest_rows: List[Dict[str, object]] = []
        per_seed_results = []
        for seed in args.seeds:
            print(f"\n### Seed {seed}")
            seed_result = run_budget_sweep(
                strategies=all_strategies,
                answer_llm=answer_llm,
                judge_llm=judge_llm,
                data_path=args.data_path,
                budgets=sorted(args.budgets),
                max_samples=args.max_samples,
                output_dir=args.output_dir,
                seed=seed,
                shuffle=args.shuffle,
                judge_mode=args.judge_mode,
                model_name=model_name,
                judge_model_name=args.judge_model,
                config_path=args.config,
                benchmark=args.benchmark,
                sample_ids=None,
                progress_callback=lambda strategy_name, result, completed_budgets, seed=seed: (
                    print(
                        f"  [checkpoint] seed={seed} strategy={strategy_name} "
                        f"budgets={completed_budgets} saved",
                        flush=True,
                    ),
                    save_progress_snapshot(
                        output_dir=args.output_dir,
                        model_name=model_name,
                        seed=seed,
                        strategy_name=strategy_name,
                        result=result,
                        completed_budgets=completed_budgets,
                    ),
                )
            )
            per_seed_results.append(seed_result)
            completed_manifest_rows.append(
                {
                    "seed": seed,
                    "strategies": sorted(seed_result.keys()),
                }
            )
            save_run_manifest(
                output_dir=args.output_dir,
                model_name=model_name,
                data_path=args.data_path,
                budgets=sorted(args.budgets),
                max_samples=args.max_samples,
                seeds=args.seeds,
                strategies=[s.name for s in all_strategies],
                completed=completed_manifest_rows,
            )

        aggregated = aggregate_runs(per_seed_results)
        all_outputs[model_name] = {
            "raw_runs": per_seed_results,
            "aggregate": aggregated,
        }
        path = os.path.join(args.output_dir, f"budget_sweep_{args.benchmark}_{model_name.replace('/', '_')}.json")
        save_json(path, all_outputs[model_name])
        print(f"\nResults saved: {path}")

    summary_path = os.path.join(args.output_dir, f"budget_sweep_summary_{args.benchmark}.json")
    save_json(summary_path, all_outputs)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
