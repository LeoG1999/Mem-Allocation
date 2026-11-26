#!/usr/bin/env python3
"""Generate Figures 2--6 from the aggregated tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(root: Path, pattern: str) -> dict:
    path = root / pattern
    return json.loads(path.read_text(encoding="utf-8"))


def curve(rows: list[dict]) -> tuple[list[int], list[float]]:
    return (
        [int(row["budget"]) for row in rows],
        [float(row["accuracy_pct"]) for row in rows],
    )


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure2(tables: Path, output: Path) -> None:
    main = load(tables, "table_01_main.json")
    extended = load(tables, "table_16_extended_budget.json")
    paired = load(tables, "table_07_paired_statistics.json")
    split_rows = list(main["deepseek"]["dual_track"])
    track_rows = list(main["deepseek_trackb"]["bm25_chunk_trackb"])
    for budget in (12000, 24000):
        split_rows.append({"budget": budget, **extended["dual_track_split"][str(budget)]})
        track_rows.append({"budget": budget, **extended["track_b_concentration"][str(budget)]})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].axvspan(12000, 24000, color="0.92", zorder=0)
    for rows, label in ((split_rows, "Dual-Track Split"), (track_rows, "Track-B concentration")):
        x, y = curve(rows)
        axes[0].plot(x, y, marker="o", label=label)
    full = extended.get("full_history", {}).get("12000", {}).get("accuracy_pct")
    if full is not None:
        axes[0].axhline(full, linestyle=":", color="black", label="Full history")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Budget (characters)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=8)
    rows = paired["comparisons"]["dual_track_vs_bm25_trackb_full"]
    x = [row["budget"] for row in rows]
    y = [100 * row["delta"] for row in rows]
    low = [value - 100 * row["ci95_low"] for value, row in zip(y, rows)]
    high = [100 * row["ci95_high"] - value for value, row in zip(y, rows)]
    axes[1].errorbar(x, y, yerr=[low, high], marker="o", capsize=3)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Budget (characters)")
    axes[1].set_ylabel("Split − Track-B (pp)")
    save(fig, output / "figure_02.png")


def figure3(tables: Path, output: Path) -> None:
    across = load(tables, "table_17_allocation_across_budgets.json")
    oracle = load(tables, "table_18_allocation_oracles.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ratios = {"trackb1024": "0:100", "ratio25": "25:75", "ratio33": "33:67",
              "ratio50": "50:50", "ratio67": "67:33", "ratio75": "75:25", "tracka1024": "100:0"}
    for name, rows in across["static"].items():
        x = sorted(int(value) for value in rows)
        y = [rows[str(value)]["accuracy_pct"] for value in x]
        axes[0].plot(x, y, marker="o", label=ratios[name])
    axes[0].set_xlabel("Budget (characters)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=7, ncol=2)
    static4 = oracle["static"]
    policies = [name for name in ratios if "4000" in static4.get(name, {})]
    labels = [ratios[name] for name in policies]
    values = [static4[name]["4000"]["accuracy_pct"] for name in policies]
    if oracle.get("router"):
        labels.append("heuristic replay")
        values.append(oracle["router"]["accuracy_pct"])
    if oracle.get("five_ratio_oracle", {}).get("4000"):
        labels.append("five-ratio oracle")
        values.append(oracle["five_ratio_oracle"]["4000"]["accuracy_pct"])
    axes[1].bar(range(len(values)), values)
    axes[1].set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    axes[1].set_ylabel("Accuracy at 4k (%)")
    save(fig, output / "figure_03.png")


def figure4(tables: Path, output: Path) -> None:
    table = load(tables, "table_13_masking.json")["conditions"]
    tasks = ["overall", "single_session_user", "single_session_preference"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.25
    conditions = {"dual_track_split": "Dual-Track Split", "chunks_only_split_budget": "Track B only",
                  "summaries_only_split_budget": "Track A only"}
    for offset, (condition, label) in enumerate(conditions.items()):
        row = table[condition]
        values = [row["accuracy_pct"]]
        for task in tasks[1:]:
            values.append(row["by_task"][task]["accuracy_pct"])
        x = [index + (offset - 1) * width for index in range(len(tasks))]
        ax.bar(x, values, width=width, label=label)
    ax.set_xticks(range(len(tasks)), ["Overall", "Single-session user", "Preference"])
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7)
    save(fig, output / "figure_04.png")


def figure5(tables: Path, output: Path) -> None:
    table = load(tables, "table_23_locomo.json")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for source, strategy, label in (
        ("split", "dual_track", "Dual-Track Split"),
        ("trackb", "bm25_chunk_trackb", "Track-B concentration"),
    ):
        x, y = curve(table[source][strategy])
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("Budget (characters)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    save(fig, output / "figure_05.png")


def figure6(tables: Path, output: Path) -> None:
    table = load(tables, "table_14_span_coverage.json")["by_task_type"]
    tasks = list(table)
    keys = ["a_only_pct", "b_only_pct", "both_pct", "neither_pct", "unknown_pct"]
    labels = ["A only", "B only", "Both", "Neither", "Unknown"]
    bottom = [0.0] * len(tasks)
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label in zip(keys, labels):
        values = [100 * float(table[task].get(key, 0)) for task in tasks]
        ax.bar(tasks, values, bottom=bottom, label=label)
        bottom = [left + right for left, right in zip(bottom, values)]
    ax.set_ylabel("Coverage (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=7, ncol=5)
    save(fig, output / "figure_06.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=Path("data/results/paper_tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/paper_figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure2(args.tables_dir, args.output_dir)
    figure3(args.tables_dir, args.output_dir)
    figure4(args.tables_dir, args.output_dir)
    figure5(args.tables_dir, args.output_dir)
    figure6(args.tables_dir, args.output_dir)
    print(f"Generated Figures 2--6 in {args.output_dir}")


if __name__ == "__main__":
    main()
