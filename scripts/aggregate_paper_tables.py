#!/usr/bin/env python3
"""Convert experiment outputs into JSON files for the paper tables."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return round(statistics.fmean(numbers), 4) if numbers else None


def summarize_rows(rows: list[dict]) -> dict:
    correct = sum(bool(row.get("is_correct")) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy_pct": round(100 * correct / len(rows), 3) if rows else None,
        "avg_input_tokens": mean([row.get("input_tokens") for row in rows]),
        "avg_context_chars": mean([row.get("memory_context_len") for row in rows]),
    }


def sweep_table(path: Path, cohort: set) -> dict:
    payload = read_json(path)
    run = payload["raw_runs"][0] if payload.get("raw_runs") else payload
    table = {}
    for strategy, result in run.items():
        if not isinstance(result, dict) or "budgets" not in result:
            continue
        if len(set(result["budgets"])) != len(result["budgets"]):
            raise ValueError(f"Duplicate budget for {strategy} in {path}")
        rows = []
        correctness = result.get("sample_correctness", [])
        for index, budget in enumerate(result["budgets"]):
            sample_rows = correctness[index] if index < len(correctness) else []
            accuracy = result.get("accuracies", [None] * len(result["budgets"]))[index]
            tokens = result.get("avg_input_tokens", [None] * len(result["budgets"]))[index]
            if sample_rows:
                ids = {r["sample_id"] for r in sample_rows}
                if len(ids) != len(sample_rows) or (cohort and ids != cohort):
                    raise ValueError(f"Inconsistent sample IDs for {strategy}/{budget} in {path}")
                cohort.update(ids)
                flags = [r["is_correct"] for r in sample_rows]
                if not all(isinstance(v, bool) for v in flags):
                    raise ValueError(f"Invalid correctness label for {strategy}/{budget} in {path}")
                accuracy = statistics.mean(flags)
                if all(r.get("input_tokens") is not None for r in sample_rows):
                    tokens = statistics.mean(r["input_tokens"] for r in sample_rows)
            rows.append({
                "budget": budget,
                "n": len(sample_rows),
                "accuracy_pct": round(100 * float(accuracy), 3) if accuracy is not None else None,
                "avg_input_tokens": tokens,
            })
        table[strategy] = rows
    return table


def jsonl_cells(raw_dir: Path, cohort: set) -> dict:
    cells = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        if rows:
            ids = {r["sample_id"] for r in rows}
            if len(ids) != len(rows) or (cohort and ids != cohort):
                raise ValueError(f"Inconsistent sample IDs in {path}")
            if not all(isinstance(r["is_correct"], bool) for r in rows):
                raise ValueError(f"Invalid correctness label in {path}")
            cells[path.stem] = summarize_rows(rows)
    return cells


def write_table(output_dir: Path, number: int, name: str, payload: Any) -> None:
    path = output_dir / f"table_{number:02d}_{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/paper_tables"))
    parser.add_argument("--with-appendix", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    results = args.results_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing = []

    sources = {
        "deepseek": results / "main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json",
        "deepseek_trackb": results / "trackb_deepseek/trackb_fullbudget_results.json",
        "qwen": results / "main_qwen/budget_sweep_longmemeval_Qwen_Qwen3.5-9B.json",
        "qwen_trackb": results / "trackb_qwen/trackb_fullbudget_results.json",
    }
    main_table = {}
    long_ids = set()
    for name, path in sources.items():
        if path.exists():
            main_table[name] = sweep_table(path, long_ids)
        else:
            missing.append(str(path))
    expected_main = {
        "deepseek": ("recent_sessions", "bm25_session", "session_summary", "embedding_chunk", "bm25_chunk", "dual_track"),
        "deepseek_trackb": ("bm25_chunk_trackb",),
        "qwen": ("recent_sessions", "bm25_session", "session_summary", "embedding_chunk", "bm25_chunk", "dual_track"),
        "qwen_trackb": ("bm25_chunk_trackb",),
    }
    for source, strategies in expected_main.items():
        for strategy in strategies:
            rows = main_table.get(source, {}).get(strategy, [])
            if {r["budget"] for r in rows} != {1000, 2000, 4000, 6000} or any(r["n"] != 500 for r in rows):
                missing.append(f"{source}:{strategy} requires four n=500 cells")
    write_table(args.output_dir, 1, "main", main_table)
    token_table = {}
    deepseek_split = {row["budget"]: row for row in main_table.get("deepseek", {}).get("dual_track", [])}
    deepseek_track = {
        row["budget"]: row
        for row in main_table.get("deepseek_trackb", {}).get("bm25_chunk_trackb", [])
    }
    for budget in sorted(set(deepseek_split) & set(deepseek_track)):
        split_tokens = deepseek_split[budget]["avg_input_tokens"]
        track_tokens = deepseek_track[budget]["avg_input_tokens"]
        token_table[str(budget)] = {
            "dual_track_split": split_tokens,
            "track_b_concentration": track_tokens,
            "split_reduction_pct": round(100 * (track_tokens - split_tokens) / track_tokens, 3),
        }
    for strategy in ("bm25_chunk", "embedding_chunk"):
        rows = main_table.get("deepseek", {}).get(strategy, [])
        match = next((row for row in rows if row["budget"] == 4000), None)
        if match:
            token_table[f"{strategy}_4k"] = match["avg_input_tokens"]
    write_table(args.output_dir, 5, "input_tokens", token_table)

    controls_path = results / "paper_controls/tables.json"
    controls = read_json(controls_path) if controls_path.exists() else {}
    if not controls:
        missing.append(str(controls_path))
    for item in controls.get("missing_requirements", []):
        missing.append(f"paper_controls: {item}")
    controls_map = {
        2: ("chunk_size", "table_chunk_size_control"),
        6: ("budget_utilization", "table_budget_utilization"),
        10: ("allocation_judge", "table_main_judge_audit_grid"),
        11: ("question_inclusive_judge", "table_independent_and_question_inclusive_judges"),
        15: ("win_loss_coverage", "table_win_loss_coverage"),
        19: ("router_task_matrix", "table_router_task_matrix"),
    }
    for number, (name, key) in controls_map.items():
        write_table(args.output_dir, number, name, controls.get(key, {}))
    static_allocation = controls.get("table_static_allocation", {})
    write_table(args.output_dir, 17, "allocation_across_budgets", {
        "static": {
            key: static_allocation.get(key, {})
            for key in ("trackb1024", "ratio25", "ratio33", "ratio50", "tracka1024")
        },
        "router": controls.get("table_router_across_budgets", {}),
    })
    write_table(args.output_dir, 18, "allocation_oracles", {
        "static": static_allocation,
        "router": controls.get("table_router_across_budgets", {}).get("4000", {}),
        "three_ratio_oracle": controls.get("table_action_matched_oracle", {}),
        "five_ratio_oracle": controls.get("table_five_ratio_oracle", {}),
    })

    dense_dir = results / "dense_matched/raw"
    dense = jsonl_cells(dense_dir, long_ids)
    if not dense:
        missing.append(str(dense_dir))
    for condition in ("dense_split", "dense_trackb"):
        for budget in (1000, 2000, 4000, 6000):
            key = f"{condition}_b{budget}"
            if dense.get(key, {}).get("n") != 500:
                missing.append(f"dense matched {condition} b={budget} requires n=500")
    length_matched = controls.get("table_length_matched_concentration", {})
    for budget in sorted(deepseek_split.keys() & deepseek_track.keys()):
        split = deepseek_split[budget]["accuracy_pct"]
        chunks = deepseek_track[budget]["accuracy_pct"]
        dense[f"bm25_split_b{budget}"] = deepseek_split[budget]
        dense[f"bm25_trackb_b{budget}"] = deepseek_track[budget]
        dense[f"bm25_delta_pp_b{budget}"] = round(split - chunks, 3)
        split_cell = dense.get(f"dense_split_b{budget}")
        chunk_cell = dense.get(f"dense_trackb_b{budget}")
        if split_cell and chunk_cell:
            dense[f"dense_delta_pp_b{budget}"] = round(split_cell["accuracy_pct"] - chunk_cell["accuracy_pct"], 3)
        row = length_matched.get(str(budget))
        if row:
            row.update(split_accuracy_pct=split, full_cap_accuracy_pct=chunks,
                       split_minus_length_matched_pp=round(split - row["accuracy_pct"], 3))
        else:
            missing.append(f"length-matched b={budget} requires a measured accuracy")
    write_table(args.output_dir, 12, "length_matched", length_matched)
    write_table(args.output_dir, 3, "dense_matched", dense)

    paired = results / "statistics/paired_crossover.json"
    paired_payload = read_json(paired) if paired.exists() else {}
    write_table(args.output_dir, 7, "paired_statistics", paired_payload)
    if not paired.exists():
        missing.append(str(paired))
    for row in paired_payload.get("comparisons", {}).get("dual_track_vs_bm25_trackb_full", []):
        if row.get("n") != 500:
            missing.append(f"paired comparison b={row.get('budget')} requires n=500")

    judge = results / "judge_validation_full/matched_control_judge_validation.json"
    judge_payload = read_json(judge) if judge.exists() else {}
    write_table(args.output_dir, 8, "independent_judge", judge_payload)
    if not judge.exists():
        missing.append(str(judge))
    for budget in (1000, 2000, 4000, 6000):
        row = judge_payload.get("summary_by_budget", {}).get(str(budget), {})
        if row.get("independent_judge", {}).get("n_shared") != 500:
            missing.append(f"independent judge b={budget} requires 500 shared items")
        if row.get("cross_judge", {}).get("n_decisions") != 1000:
            missing.append(f"cross-judge agreement b={budget} requires 1000 decisions")

    subset = results / "judge_validation_matched_control/matched_control_judge_validation.json"
    subset_audit = read_json(subset) if subset.exists() else {}
    write_table(args.output_dir, 9, "matched_subset_audit", subset_audit)
    for budget in (1000, 4000, 6000):
        row = subset_audit.get("summary_by_budget", {}).get(str(budget), {})
        if row.get("independent_judge", {}).get("n_shared") != 100 or row.get("cross_judge", {}).get("n_decisions") != 200:
            missing.append(f"subset audit b={budget} requires 100 paired questions")

    masking = results / "masking/table13.json"
    if not masking.exists() and (results / "masking/table12.json").is_file():
        masking = results / "masking/table12.json"
    masking_payload = read_json(masking) if masking.exists() else {}
    write_table(args.output_dir, 13, "masking", masking_payload)
    if not masking.exists():
        missing.append(str(masking))
    for condition, row in masking_payload.get("conditions", {}).items():
        if row.get("n") != 500:
            missing.append(f"masking {condition} requires n=500")

    coverage = results / "coverage/table14.json"
    if not coverage.exists() and (results / "coverage/table13.json").is_file():
        coverage = results / "coverage/table13.json"
    write_table(args.output_dir, 14, "span_coverage", read_json(coverage) if coverage.exists() else {})
    if not coverage.exists():
        missing.append(str(coverage))

    extended_dir = results / "extended_budget/raw"
    extended = jsonl_cells(extended_dir, long_ids)
    if not extended:
        missing.append(str(extended_dir))
    extended_table = {
        "dual_track_split": {
            str(budget): extended.get(f"split_extended_b{budget}", {})
            for budget in (12000, 24000)
        },
        "track_b_concentration": {
            str(budget): extended.get(f"trackb1024_b{budget}", {})
            for budget in (12000, 24000)
        },
        "full_history": {
            str(budget): extended.get("full_history_b12000", {})
            for budget in (12000, 24000)
        },
    }
    for condition in ("dual_track_split", "track_b_concentration"):
        for budget in (12000, 24000):
            if extended_table[condition][str(budget)].get("n") != 500:
                missing.append(f"extended {condition} b={budget} requires n=500")
    if extended_table["full_history"]["12000"].get("n") != 500:
        missing.append("full-history reference requires n=500")
    write_table(args.output_dir, 16, "extended_budget", extended_table)

    locomo_sources = {
        "split": results / "locomo_split/budget_sweep_locomo_deepseek-v4-flash.json",
        "trackb": results / "locomo_trackb/budget_sweep_locomo_deepseek-v4-flash.json",
        "references": results / "locomo_references/budget_sweep_locomo_deepseek-v4-flash.json",
    }
    locomo = {}
    locomo_ids = set()
    for name, path in locomo_sources.items():
        if path.exists():
            locomo[name] = sweep_table(path, locomo_ids)
        else:
            missing.append(str(path))
    expected_locomo = {
        "split": {"dual_track": (500, 1000, 2000, 4000)},
        "trackb": {"bm25_chunk_trackb": (500, 1000, 2000, 4000)},
        "references": {
            "recent_sessions": (1000, 2000, 4000),
            "bm25_session": (1000, 2000, 4000),
            "session_summary": (1000, 2000, 4000),
            "embedding_chunk": (1000, 2000, 4000),
        },
    }
    for source, strategies in expected_locomo.items():
        for strategy, budgets in strategies.items():
            rows = locomo.get(source, {}).get(strategy, [])
            actual = {row.get("budget"): row.get("n") for row in rows}
            if any(actual.get(budget) != 200 for budget in budgets):
                missing.append(f"LoCoMo {source}:{strategy} requires n=200 at {list(budgets)}")
    write_table(args.output_dir, 23, "locomo", locomo)

    generated = [1, 2, 3, *range(5, 20), 23]
    if args.with_appendix:
        appendix = results / "appendix"
        selectors = appendix / "selectors.json"
        if selectors.exists():
            replay = read_json(selectors)
            decisions = replay["out_of_fold"]
            ids = {r["sample_id"] for r in decisions}
            if len(ids) != 500 or len(decisions) != 500 or (long_ids and ids != long_ids):
                missing.append("selector replay requires 500 distinct out-of-fold decisions")
            write_table(args.output_dir, 20, "selectors", replay["summary"])
            allocation_path = args.output_dir / "table_18_allocation_oracles.json"
            allocation = read_json(allocation_path)
            allocation["learned_selector"] = {"n": len(decisions), "accuracy_pct": replay["summary"]["main"]["learned"]}
            write_table(args.output_dir, 18, "allocation_oracles", allocation)
            generated.append(20)
        else:
            missing.append(str(selectors))
        for number, name, budgets in [(21, "joint", {1000, 2000, 4000, 6000}), (22, "views", {1000, 4000})]:
            path = appendix / name / "summary.json"
            if not path.exists():
                missing.append(str(path))
                continue
            rows = read_json(path)
            policies = {"split", "matched"} if name == "joint" else {
                "chunks", "extract", "summary", "facts", "extract+chunks", "summary+chunks+facts",
            }
            judge_field = "independent_correct" if name == "joint" else "is_correct"
            if ({r["budget"] for r in rows} != budgets or len(rows) != len(budgets)
                    or any(set(r["policies"]) != policies or r["judge"] != judge_field for r in rows)
                    or any(p["n"] != 500 for r in rows for p in r["policies"].values())):
                missing.append(f"{name} requires a complete 500-item grid")
            write_table(args.output_dir, number, name, rows)
            generated.append(number)
    manifest = {
        "generated_tables": sorted(generated),
        "table_4": "Method definition; no experimental values",
        "missing_sources": sorted(set(missing)),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Generated table JSON files; {len(manifest['missing_sources'])} input(s) missing")
    if manifest["missing_sources"] and not args.allow_missing:
        raise SystemExit("Aggregation stopped because required experiment artifacts are incomplete")


if __name__ == "__main__":
    main()
