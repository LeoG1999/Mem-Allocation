#!/usr/bin/env python3
"""Offline aggregation for artifacts produced by exp_paper_controls.py.

This script performs no model calls. Missing prerequisite cells remain absent
and are listed under ``missing_requirements``; they are never imputed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp_orthogonality import check_coverage, stored_track_contexts
from src.memory.track_b_rag import payload_chars


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{line_no}") from exc
    return rows


def pct(correct: int, n: int) -> float | None:
    return round(100.0 * correct / n, 3) if n else None


def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return round(float(ordered[index]), 4)


def index_raw(run_dir: Path) -> dict[tuple[str, int], list[dict]]:
    cells = {}
    for path in sorted((run_dir / "raw").glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        key = (str(rows[0]["condition"]), int(rows[0]["nominal_budget_chars"]))
        if any((r["condition"], int(r["nominal_budget_chars"])) != key for r in rows):
            raise ValueError(f"Mixed condition/budget rows in {path}")
        ids = [r["sample_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in {path}")
        cells[key] = rows
    return cells


def cell_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "correct": sum(bool(r["is_correct"]) for r in rows),
        "accuracy_pct": pct(sum(bool(r["is_correct"]) for r in rows), len(rows)),
        "avg_context_chars": mean(r.get("memory_context_len") for r in rows),
        "avg_answer_input_tokens": mean(r.get("input_tokens") for r in rows),
        "api_errors": sum(r.get("prediction") == "[API_ERROR]" for r in rows),
    }


def chunk_table(cells: dict) -> dict:
    table = {}
    for chunk in (256, 512, 1024):
        split_name = "ratio50" if chunk == 1024 and ("ratio50", 1000) in cells else f"split{chunk}"
        track_name = f"trackb{chunk}"
        by_budget = {}
        for budget in (1000, 2000, 4000, 6000):
            split = cells.get((split_name, budget))
            track = cells.get((track_name, budget))
            if not split or not track:
                continue
            s = cell_summary(split)
            t = cell_summary(track)
            by_budget[str(budget)] = {
                "split": s,
                "track_b": t,
                "delta_pp": round(s["accuracy_pct"] - t["accuracy_pct"], 3),
            }
        if by_budget:
            table[str(chunk)] = by_budget
    return table


def allocation_tables(cells: dict, route_rows: list[dict]) -> tuple[dict, dict, dict, dict]:
    condition_names = ["trackb1024", "ratio25", "ratio33", "ratio50", "ratio67", "ratio75", "tracka1024"]
    static = {}
    for condition in condition_names:
        row = {}
        for budget in (1000, 2000, 4000, 6000):
            if (condition, budget) in cells:
                row[str(budget)] = cell_summary(cells[(condition, budget)])
        if row:
            static[condition] = row

    routes = {r["sample_id"]: r for r in route_rows}
    routed = {}
    oracle = {}
    five_ratio_oracle = {}
    for budget in (1000, 2000, 4000, 6000):
        action_cells = {
            name: {r["sample_id"]: r for r in cells.get((name, budget), [])}
            for name in ("ratio25", "ratio50", "ratio75")
        }
        common = sorted(set(routes) & set.intersection(*(set(v) for v in action_cells.values()))) if routes and all(action_cells.values()) else []
        if not common:
            continue
        route_correct = []
        oracle_correct = []
        for sid in common:
            selected = routes[sid]["selected_condition"]
            route_correct.append(bool(action_cells[selected][sid]["is_correct"]))
            oracle_correct.append(any(bool(action_cells[name][sid]["is_correct"]) for name in action_cells))
        if budget != 1000:
            routed[str(budget)] = {
                "n": len(common),
                "correct": sum(route_correct),
                "accuracy_pct": pct(sum(route_correct), len(common)),
                "parse_failures": sum(not bool(routes[sid].get("parse_success")) for sid in common),
            }
        if budget == 4000:
            oracle[str(budget)] = {
                "n": len(common),
                "correct": sum(oracle_correct),
                "accuracy_pct": pct(sum(oracle_correct), len(common)),
            }
        five_cells = {
            name: {r["sample_id"]: r for r in cells.get((name, budget), [])}
            for name in ("ratio25", "ratio33", "ratio50", "ratio67", "ratio75")
        }
        if budget == 4000 and all(five_cells.values()):
            five_common = sorted(set.intersection(*(set(v) for v in five_cells.values())))
            correct = sum(
                any(bool(five_cells[name][sid]["is_correct"]) for name in five_cells)
                for sid in five_common
            )
            five_ratio_oracle[str(budget)] = {
                "n": len(five_common), "correct": correct,
                "accuracy_pct": pct(correct, len(five_common)),
            }
    return static, routed, oracle, five_ratio_oracle


def route_task_matrix(route_rows: list[dict]) -> dict:
    matrix = defaultdict(Counter)
    for row in route_rows:
        counts = matrix[row["task_type"]]
        label = row["label"]
        counts[label] += 1
        if label == "MIXED":
            if row.get("parse_success") is True:
                counts["valid_mixed"] += 1
            elif row.get("parse_success") is False:
                counts["parse_fallbacks"] += 1
    labels = ("EVIDENCE", "STATE", "MIXED")
    return {
        task: {
            **{label: counts[label] for label in labels},
            "total": sum(counts[label] for label in labels),
            "valid_mixed": counts["valid_mixed"],
            "parse_fallbacks": counts["parse_fallbacks"],
        }
        for task, counts in sorted(matrix.items())
    }


def utilization_table(cells: dict) -> dict:
    table = {}
    for budget in (1000, 2000, 4000, 6000):
        split = cells.get(("ratio50", budget)) or cells.get(("split1024", budget))
        track = cells.get(("trackb1024", budget))
        if not split or not track:
            continue
        split_ids = {r["sample_id"] for r in split}
        if len(split_ids) != len(split) or len(split) != len(track) or split_ids != {r["sample_id"] for r in track}:
            raise ValueError(f"Table 6 requires identical paired questions at budget {budget}")
        split_injected, a_fill, b_fill, a_units, b_units = [], [], [], [], []
        cap_binding = []
        for row in split:
            attr = row.get("attribution") or row.get("packing_stats") or {}
            half = budget // 2
            caps = (attr.get("track_a_cap_chars", half), attr.get("track_b_cap_chars", budget - half))
            if (attr.get("use_router") or attr.get("mask_track_a") or attr.get("mask_track_b")
                    or caps != (half, budget - half)):
                raise ValueError(f"Table 6 requires unmasked static 50:50 data for {row['sample_id']}")
            a = attr.get("track_a_packing") or {}
            b = attr.get("track_b_packing") or {}
            a_budgeted = int(a.get("serialized_chars", len(attr.get("track_a_context") or "")))
            b_budgeted = payload_chars(b, attr.get("track_b_context") or "")
            separator = 2 if a_budgeted and b_budgeted else 0
            split_injected.append(a_budgeted + b_budgeted + separator)
            # Each track has half the nominal budget; B's packer can reserve two characters for the separator.
            a_fill.append(a_budgeted / (budget // 2))
            b_fill.append((b_budgeted + separator) / (budget - budget // 2))
            if a.get("selected_unit_count") is None or b.get("selected_unit_count") is None:
                raise ValueError(f"Missing Table 6 packing counts for {row['sample_id']} at budget {budget}")
            a_units.append(a["selected_unit_count"])
            b_units.append(b["selected_unit_count"])
            if "count_cap_binding" in b:
                cap_binding.append(b["count_cap_binding"])
        track_injected = [
            payload_chars(r.get("packing_stats") or {}, r.get("memory_context") or "")
            for r in track
        ]
        table[str(budget)] = {
            "split_n": len(split),
            "split_count_cap_binding_items": sum(cap_binding) if len(cap_binding) == len(split) else None,
            "split_serialized_chars": mean(r["memory_context_len"] for r in split),
            "split_serialized_over_nominal": mean(r["memory_context_len"] / budget for r in split),
            "split_injected_chars": mean(split_injected),
            "split_utilization_pct": mean(100 * v / budget for v in split_injected),
            "track_a_fill": mean(a_fill),
            "track_b_fill": mean(b_fill),
            "summaries_per_item": mean(a_units),
            "chunks_per_item": mean(b_units),
            "concentration_n": len(track),
            "concentration_chunks_per_item": mean(r["packing_stats"]["selected_unit_count"] for r in track),
            "concentration_serialized_chars": mean(r["memory_context_len"] for r in track),
            "concentration_injected_chars": mean(track_injected),
            "concentration_utilization_pct": mean(100 * v / budget for v in track_injected),
        }
    return table


def coverage_table(cells: dict) -> dict:
    split = {r["sample_id"]: r for r in cells.get(("ratio50", 4000), cells.get(("split1024", 4000), []))}
    track = {r["sample_id"]: r for r in cells.get(("trackb1024", 4000), [])}
    shared = sorted(set(split) & set(track))
    groups = {
        "wins": [sid for sid in shared if split[sid]["is_correct"] and not track[sid]["is_correct"]],
        "losses": [sid for sid in shared if not split[sid]["is_correct"] and track[sid]["is_correct"]],
    }
    output = {"n_shared": len(shared)}
    for group, ids in groups.items():
        buckets = Counter()
        concentration_covers = 0
        rows = []
        for sid in ids:
            srow, trow = split[sid], track[sid]
            a, b = stored_track_contexts(srow)
            cov = check_coverage(srow["gold_answer"], a, b)
            tcov = check_coverage(srow["gold_answer"], "", trow["memory_context"])
            buckets[cov["source"]] += 1
            concentration_covers += int(tcov["b_covers"])
            rows.append({"sample_id": sid, "split_source": cov["source"], "concentration_covers": tcov["b_covers"]})
        output[group] = {
            "n": len(ids),
            "counts": {name: buckets[name] for name in ("A_only", "B_only", "both", "neither", "unknown")},
            "percentages": {name: pct(buckets[name], len(ids)) for name in ("A_only", "B_only", "both", "neither", "unknown")},
            "concentration_covers_count": concentration_covers,
            "concentration_covers_pct": pct(concentration_covers, len(ids)),
            "samples": rows,
        }
    output["ties"] = len(shared) - len(groups["wins"]) - len(groups["losses"])
    return output


def matched_subset_decisions(audit_dir: Path, cells: dict, audit_ids: set[str]) -> dict:
    """Read the two overlapping Table 10 cells from the Table 9 judgments."""
    decisions = {}
    for condition, strategy in (("ratio50", "dual_track"), ("trackb1024", "bm25_chunk_trackb")):
        decisions[condition] = {}
        path = audit_dir / "_partial" / f"matched_judge_{strategy}_b4000_gpt-4o-mini.json"
        if not path.exists():
            continue
        audit = json.loads(path.read_text(encoding="utf-8"))
        if (audit["strategy"] != strategy or audit["budget"] != 4000
                or audit["judging"]["judge_model"] != "gpt-4o-mini"):
            raise ValueError(f"Incompatible matched audit: {path}")
        source = {r["sample_id"]: r for r in cells.get((condition, 4000), [])}
        for row in audit["samples"]:
            sid = row["sample_id"]
            original = source.get(sid)
            if (sid in decisions[condition] or sid not in audit_ids or original is None
                    or any(row[k] != original[k] for k in ("prediction", "gold_answer", "task_type"))
                    or ("question" in row and row["question"] != original["question"])
                    or ("main_judge_correct" in row and row["main_judge_correct"] != original["is_correct"])
                    or not isinstance(row["is_correct"], bool)):
                raise ValueError(f"Matched audit differs from {condition} for {sid}")
            decisions[condition][sid] = row["is_correct"]
        if decisions[condition].keys() != audit_ids:
            raise ValueError(f"Incomplete matched audit: {path}")
    return decisions


def main_judge_audit_grid(
    cells: dict,
    audit_ids: set[str],
    run_dir: Path,
    matched_decisions=None,
) -> dict:
    if not audit_ids:
        return {}
    conditions = (
        "trackb1024", "ratio25", "ratio33", "ratio50",
        "ratio67", "ratio75", "tracka1024", "router",
    )
    output = {}
    independent_maps = defaultdict(dict)
    for path in sorted((run_dir / "rejudge").glob("*.jsonl")):
        for row in read_jsonl(path):
            if (
                row.get("prompt_mode") == "questionless"
                and row.get("judge_model") == "gpt-4o-mini"
                and int(row.get("nominal_budget_chars") or 0) == 4000
                and row.get("sample_id") in audit_ids
            ):
                independent_maps[row.get("condition")][row["sample_id"]] = bool(row["is_correct"])
    if matched_decisions is not None:
        independent_maps.update(matched_decisions)
    for condition in conditions:
        rows = cells.get((condition, 4000), [])
        subset = [r for r in rows if r["sample_id"] in audit_ids]
        if subset:
            output[condition] = {
                "full_set": cell_summary(rows),
                "main_judge_audit_subset": cell_summary(subset),
                "main_judge_model": subset[0].get("main_judge_model"),
            }
            independent = independent_maps.get(condition, {})
            shared = sorted(audit_ids & set(independent))
            output[condition]["independent_judge_audit_subset"] = {
                "n": len(shared),
                "correct": sum(independent[sid] for sid in shared),
                "accuracy_pct": pct(sum(independent[sid] for sid in shared), len(shared)),
                "judge_model": "gpt-4o-mini",
            }
    return output


def length_table(run_dir: Path) -> dict:
    output = {}
    for path in sorted((run_dir / "length_matched").glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        budget = str(rows[0]["nominal_budget_chars"])
        correct = sum(bool(r["is_correct"]) for r in rows)
        mismatch = [float(r["absolute_mismatch_chars"]) for r in rows]
        rel = [float(r["relative_mismatch"]) for r in rows]
        output[budget] = {
            "n": len(rows),
            "correct": correct,
            "accuracy_pct": pct(correct, len(rows)),
            "mean_absolute_mismatch_chars": mean(mismatch),
            "median_absolute_mismatch_chars": quantile(mismatch, 0.5),
            "p90_absolute_mismatch_chars": quantile(mismatch, 0.9),
            "mean_relative_mismatch": mean(rel),
        }
    return output


def rejudge_tables(run_dir: Path, route_rows: list[dict], matched_decisions: dict) -> dict:
    groups = defaultdict(list)
    decisions = defaultdict(dict)
    verdict_maps = defaultdict(dict)
    source_main_maps = defaultdict(dict)
    for path in sorted((run_dir / "rejudge").glob("*.jsonl")):
        rows = read_jsonl(path)
        for row in rows:
            key = (
                row.get("judge_model"), row.get("prompt_mode"), row.get("condition"),
                int(row.get("nominal_budget_chars") or 0),
            )
            groups[key].append(row)
            verdict_maps[key][row["sample_id"]] = bool(row["is_correct"])
            if row.get("source_main_correct") is not None:
                source_main_maps[key][row["sample_id"]] = bool(row["source_main_correct"])
            decision_key = (
                row.get("judge_model"), row.get("condition"),
                int(row.get("nominal_budget_chars") or 0), row["sample_id"],
                row.get("source_prediction_sha256"),
            )
            decisions[decision_key][row.get("prompt_mode")] = bool(row["is_correct"])
    summary = {}
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        model, prompt, condition, budget = key
        correct = sum(bool(r["is_correct"]) for r in rows)
        summary[f"{model}|{prompt}|{condition}|{budget}"] = {
            "n": len(rows), "correct": correct, "accuracy_pct": pct(correct, len(rows)),
            "parse_failures": sum(not r.get("parse_success", False) for r in rows),
        }
    for condition, labels in matched_decisions.items():
        key = ("gpt-4o-mini", "questionless", condition, 4000)
        verdict_maps[key] = labels
        if labels:
            correct = sum(labels.values())
            summary["|".join(map(str, key))] = {
                "n": len(labels), "correct": correct, "accuracy_pct": pct(correct, len(labels)),
            }
    agreement_groups = defaultdict(list)
    for key, modes in decisions.items():
        if "questionless" in modes and "question_inclusive" in modes:
            model, condition, budget, _, _ = key
            agreement_groups[(model, condition, budget)].append(
                modes["questionless"] == modes["question_inclusive"]
            )
    agreement = {
        f"{model}|{condition}|{budget}": {
            "n": len(vals),
            "agreement_pct": pct(sum(vals), len(vals)),
        }
        for (model, condition, budget), vals in sorted(agreement_groups.items())
    }
    routes = {row["sample_id"]: row for row in route_rows}
    composed_router = {}
    model_prompt_budgets = sorted({(key[0], key[1], key[3]) for key in verdict_maps})
    for model, prompt, budget in model_prompt_budgets:
        actions = {
            condition: verdict_maps.get((model, prompt, condition, budget), {})
            for condition in ("ratio25", "ratio50", "ratio75")
        }
        if not routes or not all(actions.values()):
            continue
        common = sorted(set(routes) & set.intersection(*(set(v) for v in actions.values())))
        vals = [actions[routes[sid]["selected_condition"]][sid] for sid in common]
        composed_router[f"{model}|{prompt}|{budget}"] = {
            "n": len(vals), "correct": sum(vals), "accuracy_pct": pct(sum(vals), len(vals)),
            "composed_from_stored_action_verdicts": True,
        }

    prompt_sensitivity = {}
    models = sorted({key[0] for key in verdict_maps})
    budgets = sorted({key[3] for key in verdict_maps})
    for model in models:
        for budget in budgets:
            q_cells = {
                condition: verdict_maps.get((model, "question_inclusive", condition, budget), {})
                for condition in ("ratio50", "trackb1024")
            }
            if not all(q_cells.values()):
                continue
            q_common = sorted(set(q_cells["ratio50"]) & set(q_cells["trackb1024"]))
            split_correct = sum(q_cells["ratio50"][sid] for sid in q_common)
            track_correct = sum(q_cells["trackb1024"][sid] for sid in q_common)
            paired_decisions = []
            for condition in ("ratio50", "trackb1024"):
                new = verdict_maps.get((model, "question_inclusive", condition, budget), {})
                main = source_main_maps.get((model, "question_inclusive", condition, budget), {})
                paired_decisions.extend(main[sid] == new[sid] for sid in sorted(set(main) & set(new)))
            prompt_sensitivity[f"{model}|{budget}"] = {
                "n_items_per_policy": len(q_common),
                "split_accuracy_pct": pct(split_correct, len(q_common)),
                "track_b_accuracy_pct": pct(track_correct, len(q_common)),
                "delta_pp": round(
                    pct(split_correct, len(q_common)) - pct(track_correct, len(q_common)), 3
                ) if q_common else None,
                "n_paired_prompt_decisions": len(paired_decisions),
                "main_vs_question_inclusive_agreement_pct": pct(
                    sum(paired_decisions), len(paired_decisions)
                ),
            }
    return {
        "accuracy_cells": summary,
        "prompt_agreement": agreement,
        "question_inclusive_pair_table": prompt_sensitivity,
        "composed_router_audit": composed_router,
    }


def missing_requirements(
    cells: dict,
    route_rows: list[dict],
    run_dir: Path,
    audit_ids: set[str],
    matched_decisions: dict,
) -> list[str]:
    missing = []
    required = []
    for chunk in (256, 512):
        for kind in ("split", "trackb"):
            for budget in (1000, 2000, 4000, 6000):
                required.append((f"{kind}{chunk}", budget))
    for condition in ("ratio25", "ratio33", "ratio50", "trackb1024", "tracka1024"):
        for budget in (1000, 2000, 4000, 6000):
            required.append((condition, budget))
    for budget in (2000, 4000, 6000):
        required.append(("ratio75", budget))
    required.extend([("ratio67", 4000), ("router", 4000)])
    base_ids = {r["sample_id"] for r in cells.get(("ratio50", 4000), [])}
    for key in required:
        if key not in cells:
            missing.append(f"raw cell {key[0]} b={key[1]}")
        elif len(cells[key]) != 500:
            missing.append(f"raw cell {key[0]} b={key[1]} has n={len(cells[key])}, expected 500")
        elif {r["sample_id"] for r in cells[key]} != base_ids:
            missing.append(f"question IDs differ for {key}")
    route_ids = {row.get("sample_id") for row in route_rows}
    if route_ids != base_ids:
        missing.append("frozen routes and main sweep have different question IDs")
    if len(route_rows) != 500 or len(route_ids) != 500:
        missing.append(f"Route-B labels have n={len(route_rows)}, unique={len(route_ids)}, expected 500")
    if len(audit_ids) != 100:
        missing.append(f"independent-judge audit plan has n={len(audit_ids)}, expected 100")

    rejudge_counts = defaultdict(set)
    for path in sorted((run_dir / "rejudge").glob("*.jsonl")):
        for row in read_jsonl(path):
            key = (
                row.get("judge_model"), row.get("prompt_mode"), row.get("condition"),
                int(row.get("nominal_budget_chars") or 0),
            )
            rejudge_counts[key].add(row.get("sample_id"))
    for condition in (
        "trackb1024", "ratio25", "ratio33", "ratio50",
        "ratio67", "ratio75", "tracka1024", "router",
    ):
        key = ("gpt-4o-mini", "questionless", condition, 4000)
        ids = set(matched_decisions[condition]) if condition in matched_decisions else rejudge_counts[key]
        if ids != audit_ids or len(ids) != 100:
            missing.append(f"Table 10 {condition} has n={len(ids)}, expected 100")
    for condition in ("ratio50", "trackb1024"):
        for budget in (1000, 2000, 4000, 6000):
            key = ("deepseek-v4-flash", "question_inclusive", condition, budget)
            if rejudge_counts[key] != base_ids or len(rejudge_counts[key]) != 500:
                missing.append(
                    f"Table 11 {condition} b={budget} has n={len(rejudge_counts[key])}, expected 500"
                )

    length_counts = defaultdict(set)
    for path in sorted((run_dir / "length_matched").glob("*.jsonl")):
        for row in read_jsonl(path):
            budget = int(row["nominal_budget_chars"])
            sid = row["sample_id"]
            if sid in length_counts[budget]:
                raise ValueError(f"Duplicate length-matched item {sid} at budget {budget}")
            length_counts[budget].add(sid)
    for budget in (1000, 2000, 4000, 6000):
        if len(length_counts[budget]) != 500 or length_counts[budget] != base_ids:
            missing.append(
                f"Table 12 length-matched b={budget} must contain the same 500 questions as the main sweep"
            )
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="data/results/paper_controls")
    parser.add_argument("--routes", default=None)
    parser.add_argument(
        "--audit-plan",
        default="data/results/judge_validation_matched_control/audit_plan.json",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    route_path = Path(args.routes) if args.routes else run_dir / "routes" / "route_b.jsonl"
    cells = index_raw(run_dir)
    route_rows = read_jsonl(route_path)
    audit_ids = set()
    audit_path = Path(args.audit_plan) if args.audit_plan else None
    if audit_path and audit_path.exists():
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if isinstance(audit_payload, dict):
            audit_payload = audit_payload.get("sample_ids") or audit_payload.get("plan", {}).get("sample_ids", [])
        if isinstance(audit_payload, list):
            audit_ids = {str(value) for value in audit_payload}
    matched_decisions = matched_subset_decisions(audit_path.parent, cells, audit_ids) if audit_path else {}
    static, routed, oracle, five_oracle = allocation_tables(cells, route_rows)
    payload = {
        "schema_version": 1,
        "source_run_dir": str(run_dir),
        "raw_cell_summaries": {f"{c}|{b}": cell_summary(rows) for (c, b), rows in sorted(cells.items())},
        "table_chunk_size_control": chunk_table(cells),
        "table_static_allocation": static,
        "table_router_across_budgets": routed,
        "table_action_matched_oracle": oracle,
        "table_five_ratio_oracle": five_oracle,
        "table_router_task_matrix": route_task_matrix(route_rows),
        "table_budget_utilization": utilization_table(cells),
        "table_main_judge_audit_grid": main_judge_audit_grid(cells, audit_ids, run_dir, matched_decisions),
        "table_independent_and_question_inclusive_judges": rejudge_tables(run_dir, route_rows, matched_decisions),
        "table_length_matched_concentration": length_table(run_dir),
        "table_win_loss_coverage": coverage_table(cells),
        "missing_requirements": missing_requirements(cells, route_rows, run_dir, audit_ids, matched_decisions),
    }
    output = Path(args.output) if args.output else run_dir / "tables.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}; raw_cells={len(cells)} missing={len(payload['missing_requirements'])}")
    if payload["missing_requirements"] and not args.allow_missing:
        raise SystemExit("Aggregation stopped because required experiment artifacts are incomplete")


if __name__ == "__main__":
    main()
