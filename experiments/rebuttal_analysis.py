"""Offline paired statistics, query costs and grouped, out-of-fold router replay."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from experiments.exp_paper_controls import read_jsonl
from experiments.exp_rebuttal import cells as expected_cells
from experiments.rebuttal_memory import GRID
from src.evaluation.stats import bootstrap_ci, mcnemar_exact_test

ACTIONS = ("ratio25", "ratio50", "ratio75")
STATIC = tuple(f"ratio{p}" for p in GRID)


def paired(left, right):
    if not left or left.keys() != right.keys():
        raise ValueError("Paired policies must have exactly the same nonempty ID set")
    x = list(left.values())
    y = [right[k] for k in left]
    wins = sum(a and not b for a, b in zip(x, y))
    losses = sum(b and not a for a, b in zip(x, y))
    diff = [100 * (int(a) - int(b)) for a, b in zip(x, y)]
    return dict(
        n=len(x),
        wins=wins,
        losses=losses,
        ties=len(x) - wins - losses,
        delta_pp=mean(diff),
        bootstrap_ci_pp=bootstrap_ci(diff),
        p_exact=mcnemar_exact_test(x, y),
    )


def holm(family):
    previous = 0
    for rank, entry in enumerate(sorted(family, key=lambda row: row["p_exact"])):
        previous = max(previous, min(1, entry["p_exact"] * (len(family) - rank)))
        entry["p_holm"] = previous


def group_cells(rows):
    groups = defaultdict(lambda: defaultdict(dict))
    if not rows:
        raise ValueError("Empty input")
    for row in rows:
        key = (row["experiment"], row["history_level"], row["budget"])
        cell = groups[key][row["condition"]]
        if row["sample_id"] in cell:
            raise ValueError("Duplicate item within a condition")
        cell[row["sample_id"]] = row
    return groups


def scores(cell, field):
    result = {sid: r[field] for sid, r in cell.items()}
    if not all(isinstance(v, bool) for v in result.values()):
        raise ValueError(f"Expected boolean {field}")
    return result


def summarize(rows, judge):
    result, families = [], defaultdict(list)
    cohorts = {}
    for (experiment, level, budget), cells in sorted(group_cells(rows).items()):
        expected = set(expected_cells(experiment, budget))
        if experiment == "views" and set(cells).isdisjoint({"extract", "summary"}):
            expected -= {"extract", "summary"}
        if set(cells) != expected:
            raise ValueError(f"Incomplete policy grid at {experiment}/{level}/{budget}")
        first = next(iter(cells.values()))
        cohort = cohorts.setdefault(experiment, set(first))
        if set(first) != cohort:
            raise ValueError("Budgets and history levels must use the same questions")
        metrics, labels, disagreements = {}, {}, {}
        for name, cell in cells.items():
            if cell.keys() != first.keys():
                raise ValueError("Different items across policies")
            labels[name] = scores(cell, judge)
            info = {"n": len(cell), "accuracy_pct": 100 * mean(labels[name].values())}
            if experiment == "token":
                inputs = sum(
                    r["answer_input_tokens"] + r["route"].get("input_tokens", 0) for r in cell.values()
                )
                outputs = sum(
                    r["answer_output_tokens"] + r["route"].get("output_tokens", 0) for r in cell.values()
                )
                info.update(
                    reference_usd=(inputs + 4 * outputs) / 1e6,
                    median_seconds=median(r["latency_seconds"] for r in cell.values()),
                    latency_modes=sorted({r.get("latency_mode", "unspecified") for r in cell.values()}),
                )
            if judge == "independent_correct":
                disagreements[name] = paired(scores(cell, "is_correct"), labels[name])
            metrics[name] = info
        report = dict(experiment=experiment, level=level, budget=budget, judge=judge, policies=metrics)
        if disagreements:
            report["main_vs_independent"] = disagreements
        if experiment == "history":
            report["mean_source_tokens"] = mean(r["source_tokens"] for r in first.values())
        if experiment == "views":
            components = {"extract+chunks": ("extract", "chunks"),
                          "summary+chunks+facts": ("summary", "chunks", "facts")}
            report["mixture_minus_component_pp"] = {
                mix: {view: metrics[mix]["accuracy_pct"] - metrics[view]["accuracy_pct"]
                      for view in views if view in metrics}
                for mix, views in components.items()
            }
        else:
            a, b = ("split", "matched") if experiment == "joint" else ("ratio50", "ratio0")
            report["split_vs_chunks"] = paired(labels[a], labels[b])
            families[experiment, level].append(report["split_vs_chunks"])
        if "router" in cells:
            best = max(STATIC, key=lambda name: mean(scores(cells[name], "is_correct").values()))
            report.update(main_selected_static=best, router_vs_static=paired(labels["router"], labels[best]))
            if experiment == "audit" and budget == 4000:
                # Appendix G fixes 33:67 before independent rescoring.
                report["router_vs_prespecified_33"] = paired(labels["router"], labels["ratio33"])
            if experiment == "token" and metrics[best]["reference_usd"]:
                report["router_cost_change_pct"] = 100 * (
                    metrics["router"]["reference_usd"] / metrics[best]["reference_usd"] - 1
                )
        result.append(report)
    for family in families.values():
        holm(family)
    return result


def fit_scores(train_q, train_y, test_q, regularization=1.0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # The vocabulary and IDF are fit again within each training partition.
    features = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    train = features.fit_transform(train_q)
    test = features.transform(test_q)
    predictions = {}
    for action in ACTIONS:
        y = train_y[action]
        if len(set(y)) == 1:
            predictions[action] = [float(y[0])] * len(test_q)
        else:
            model = LogisticRegression(C=regularization, max_iter=500).fit(train, y)
            predictions[action] = model.predict_proba(test)[:, 1]
    return predictions


def tune_regularization(questions, labels, groups):
    from sklearn.model_selection import GroupKFold

    count = min(3, len(set(groups)))
    if count < 2:
        return 1.0
    gains = {}
    for strength in (0.1, 1.0, 10.0):
        correct = 0
        for train, valid in GroupKFold(n_splits=count).split(questions, groups=groups):
            probabilities = fit_scores(
                [questions[i] for i in train],
                {a: [labels[a][i] for i in train] for a in ACTIONS},
                [questions[i] for i in valid],
                strength,
            )
            for pos, i in enumerate(valid):
                selected = max(ACTIONS, key=lambda a: probabilities[a][pos])
                correct += labels[selected][i]
        gains[strength] = correct
    return max(gains, key=gains.get)


def replay(rows, folds, source_groups=None):
    from sklearn.model_selection import GroupKFold

    cells = group_cells(rows).get(("audit", "full", 4000), {})
    if not set(STATIC + ("router",)).issubset(cells):
        raise ValueError("Replay requires all seven static policies and frozen route labels at 4k")
    ids = sorted(cells["router"])
    if any(set(cell) != set(ids) for cell in cells.values()):
        raise ValueError("Incomplete candidate answers")
    cases = [cells["router"][sid] for sid in ids]
    groups = [
        source_groups[sid] if source_groups is not None else r["history_group"] for sid, r in zip(ids, cases)
    ]
    if folds < 2 or folds > len(set(groups)):
        raise ValueError("Need at least one distinct history group per fold")
    main = {act: scores(cells[act], "is_correct") for act in STATIC}
    independent = {act: scores(cells[act], "independent_correct") for act in STATIC}
    decisions = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=folds).split(ids, groups=groups)):
        static = max(STATIC, key=lambda act: sum(main[act][ids[i]] for i in train))
        train_questions = [cases[i]["question"] for i in train]
        train_labels = {a: [main[a][ids[i]] for i in train] for a in ACTIONS}
        strength = tune_regularization(train_questions, train_labels, [groups[i] for i in train])
        probabilities = fit_scores(
            train_questions, train_labels, [cases[i]["question"] for i in test], strength
        )
        for pos, i in enumerate(test):
            sid = ids[i]
            actions = dict(
                training_static=static,
                heuristic=f"ratio{cases[i]['selected_ratio']}",
                learned=max(ACTIONS, key=lambda a: probabilities[a][pos]),
            )
            if actions["heuristic"] not in ACTIONS:
                raise ValueError("Heuristic action outside the three-action grid")
            row = dict(
                sample_id=sid,
                group=groups[i],
                fold=fold,
                regularization_C=strength,
                actions=actions,
                route_label=cases[i]["route"]["label"],
                parse_success=cases[i]["route"]["parse_success"],
            )
            for judge, labels in [("main", main), ("independent", independent)]:
                row[judge] = {name: labels[a][sid] for name, a in actions.items()}
                row[judge]["oracle"] = any(labels[a][sid] for a in ACTIONS)
                row[judge]["always_half"] = labels["ratio50"][sid]
            decisions.append(row)
    summary = {}
    for judge in ("main", "independent"):
        summary[judge] = {name: 100 * mean(r[judge][name] for r in decisions) for name in decisions[0][judge]}
        summary[judge]["learned_vs_heuristic"] = paired(
            {r["sample_id"]: r[judge]["learned"] for r in decisions},
            {r["sample_id"]: r[judge]["heuristic"] for r in decisions},
        )
        summary[judge]["heuristic_by_route"] = {}
        for label in ("EVIDENCE", "STATE", "MIXED"):
            subset = [r for r in decisions if r["route_label"] == label]
            if subset:
                summary[judge]["heuristic_by_route"][label] = {
                    "n": len(subset),
                    "fallbacks": sum(not r["parse_success"] for r in subset),
                    "vs_half": paired(
                        {r["sample_id"]: r[judge]["heuristic"] for r in subset},
                        {r["sample_id"]: r[judge]["always_half"] for r in subset},
                    ),
                }
    return {"summary": summary, "out_of_fold": decisions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["summarize", "replay"])
    p.add_argument("answers", type=Path)
    p.add_argument("--judge", choices=["is_correct", "independent_correct"], default="is_correct")
    p.add_argument("--groups", type=Path, help="JSON mapping question ID to canonical source history")
    p.add_argument("--folds", default=5, type=int)
    args = p.parse_args()
    rows = read_jsonl(args.answers)
    result = (
        replay(rows, args.folds, json.loads(args.groups.read_text()) if args.groups else None)
        if args.command == "replay"
        else summarize(rows, args.judge)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
