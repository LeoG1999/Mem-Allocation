#!/usr/bin/env python3
"""API-free checks for the research code package."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import test_package_structure as package
import test_paper_controls as controls


def check_new_entrypoints(directory: Path) -> None:
    data = directory / "tiny.json"
    data.write_text(json.dumps([{
        "question_id": "q1",
        "question": "Where did Alice move?",
        "answer": "Berlin",
        "type": "single-session-user",
        "haystack_session_ids": ["session_1"],
        "haystack_dates": ["2024/01/01 (Mon) 10:00"],
        "haystack_sessions": [[
            {"role": "user", "content": "I moved to Berlin."},
            {"role": "assistant", "content": "I will remember that."},
        ]],
    }]), encoding="utf-8")
    cell = {
        "budgets": [1000],
        "accuracies": [0.5],
        "avg_input_tokens": [100],
        "sample_correctness": [[
            {"sample_id": "a", "is_correct": True},
            {"sample_id": "b", "is_correct": False},
        ]],
    }
    main_result = directory / "main.json"
    trackb_result = directory / "trackb.json"
    main_result.write_text(json.dumps({"raw_runs": [{
        "dual_track": cell,
        "bm25_chunk": cell,
        "embedding_chunk": cell,
    }]}), encoding="utf-8")
    trackb_result.write_text(json.dumps({"bm25_chunk_trackb": cell}), encoding="utf-8")
    commands = [
        [sys.executable, "-m", "experiments.exp_paper_controls", "generate",
         "--data-path", str(data), "--max-samples", "1", "--budgets", "1000",
         "--conditions", "ratio50", "--output-dir", str(directory / "controls"), "--dry-run"],
        [sys.executable, "-m", "experiments.exp_paper_controls", "route",
         "--data-path", str(data), "--max-samples", "1",
         "--output", str(directory / "route.jsonl"), "--dry-run"],
        [sys.executable, "-m", "experiments.exp_dense_matched",
         "--data-path", str(data), "--max-samples", "1", "--budgets", "1000",
         "--output-dir", str(directory / "dense"), "--dry-run"],
        [sys.executable, "-m", "experiments.exp_extended_budget",
         "--data-path", str(data), "--max-samples", "1", "--budgets", "12000", "24000",
         "--output-dir", str(directory / "extended"), "--dry-run"],
        [sys.executable, "scripts/aggregate_paper_controls.py",
         "--run-dir", str(directory / "empty-results"),
         "--output", str(directory / "control-tables.json"), "--allow-missing"],
        [sys.executable, "scripts/aggregate_paper_tables.py",
         "--results-dir", str(directory / "empty-results"),
         "--output-dir", str(directory / "paper-tables"), "--allow-missing"],
        [sys.executable, "scripts/paired_crossover_stats.py",
         "--main", str(main_result), "--trackb", str(trackb_result),
         "--bm25_chunk", str(main_result), "--budgets", "1000",
         "--bootstrap_reps", "20", "--output", str(directory / "paired.json")],
        [sys.executable, "scripts/judge_validation_matched_control.py",
         "--data_path", str(data), "--budgets", "1000", "--n_per_budget", "1",
         "--output_dir", str(directory / "judge-plan"), "--dry_run"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def check_figure_entrypoint(directory: Path) -> None:
    tables = directory / "figure-tables"
    figures = directory / "figures"
    tables.mkdir()

    def write(name: str, payload: dict) -> None:
        (tables / name).write_text(json.dumps(payload), encoding="utf-8")

    base_curve = [
        {"budget": budget, "accuracy_pct": accuracy}
        for budget, accuracy in ((1000, 30), (2000, 40), (4000, 48), (6000, 51))
    ]
    write("table_01_main.json", {
        "deepseek": {"dual_track": base_curve},
        "deepseek_trackb": {"bm25_chunk_trackb": base_curve},
    })
    write("table_16_extended_budget.json", {
        "dual_track_split": {"12000": {"accuracy_pct": 53}, "24000": {"accuracy_pct": 54}},
        "track_b_concentration": {"12000": {"accuracy_pct": 43}, "24000": {"accuracy_pct": 42}},
        "full_history": {"12000": {"accuracy_pct": 44}, "24000": {"accuracy_pct": 44}},
    })
    paired_rows = [
        {"budget": row["budget"], "delta": 0.01, "ci95_low": -0.01, "ci95_high": 0.03}
        for row in base_curve
    ]
    write("table_07_paired_statistics.json", {
        "comparisons": {"dual_track_vs_bm25_trackb_full": paired_rows}
    })
    static = {
        "ratio50": {str(row["budget"]): {"accuracy_pct": row["accuracy_pct"]} for row in base_curve}
    }
    write("table_17_allocation_across_budgets.json", {
        "static": static,
        "router": {"2000": {"accuracy_pct": 42}, "4000": {"accuracy_pct": 50}, "6000": {"accuracy_pct": 52}},
    })
    write("table_18_allocation_oracles.json", {
        "static": static,
        "three_ratio_oracle": {"4000": {"accuracy_pct": 54}},
        "five_ratio_oracle": {"4000": {"accuracy_pct": 56}},
    })
    by_task = {
        task: {"accuracy_pct": 50}
        for task in ("temporal", "knowledge_update", "single_session_user", "single_session_preference")
    }
    write("table_13_masking.json", {
        "conditions": {
            name: {"accuracy_pct": 50, "by_task": by_task}
            for name in ("dual_track_split", "chunks_only_split_budget", "summaries_only_split_budget")
        }
    })
    write("table_23_locomo.json", {
        "split": {"dual_track": base_curve},
        "trackb": {"bm25_chunk_trackb": base_curve},
    })
    write("table_14_span_coverage.json", {
        "by_task_type": {
            "temporal": {
                "a_only_pct": 0.1, "b_only_pct": 0.2, "both_pct": 0.3,
                "neither_pct": 0.25, "unknown_pct": 0.15,
            }
        }
    })
    subprocess.run(
        [sys.executable, "scripts/plot_paper_figures.py", "--tables-dir", str(tables),
         "--output-dir", str(figures)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert len(list(figures.glob("figure_*.png"))) == 5


def main() -> None:
    package.test_package_has_no_experiment_data()
    package.test_all_numeric_tables_have_an_entrypoint()
    package.test_required_entrypoints_exist()
    controls.test_route_parser_is_strict_and_uses_mixed_as_default()
    controls.test_whole_unit_length_matching_never_exceeds_serialized_target()
    with TemporaryDirectory() as directory:
        temp = Path(directory)
        controls.test_runner_persists_context_and_direct_packing_stats(temp)
        controls.test_dense_matched_strategies_run_on_fixture(temp)
        controls.test_matched_track_b_uses_identical_chunks_and_fixed_outer_overhead(temp)
        controls.test_candidate_depth_and_summary_inputs(temp)
        check_new_entrypoints(temp)
        check_figure_entrypoint(temp)
    print("run_checks: OK")


if __name__ == "__main__":
    main()
