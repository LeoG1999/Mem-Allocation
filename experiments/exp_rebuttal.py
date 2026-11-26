"""Generate new per-item runs for the camera-ready controls or rescore frozen answers."""

import argparse
from hashlib import sha256
import json
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
import random
from time import perf_counter

from experiments.exp_paper_controls import (
    ROUTE_B_PROMPT,
    ROUTE_LABEL_TO_CONDITION,
    append_jsonl,
    judge_prompt,
    parse_route_label,
    read_jsonl,
    utc_now,
    write_json,
)
from experiments.rebuttal_memory import (
    GRID,
    alternative_views,
    char_context,
    make_memory,
    pack,
    ranked_views,
    realized_chars,
    thin_history,
)
from src.evaluation.locomo_runner import load_locomo_samples
from src.evaluation.longmemeval_runner import (
    ANSWER_PROMPT,
    EvalSample,
    LongMemEvalRunner,
    parse_binary_verdict,
)
from scripts.check_data import sha256 as file_sha256
from src.models.llm_client import LLMClient


def digest(value):
    return sha256(value.encode()).hexdigest()


def token_counter(path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def source_history_groups(items):
    """Keep identical haystacks and questions sharing supporting sessions in one fold."""
    parents = {}
    owners = {}

    def root(qid):
        while parents[qid] != qid:
            parents[qid] = parents[parents[qid]]
            qid = parents[qid]
        return qid

    for item in items:
        qid = item["question_id"]
        if qid in parents:
            raise ValueError(f"Duplicate question ID: {qid}")
        parents[qid] = qid
        keys = [("history", tuple(sorted(item["haystack_session_ids"])))]
        keys.extend(("source", sid) for sid in item.get("answer_session_ids", ()))
        for key in keys:
            other = owners.setdefault(key, qid)
            a, b = sorted((root(qid), root(other)))
            parents[b] = a
    return {qid: root(qid) for qid in parents}


def prepare_audit(data, run_dir, output):
    """Join frozen 4k answers and route choices from the main policy grid by question ID."""
    groups = source_history_groups(json.loads(data.read_text()))
    routes = {}
    for row in read_jsonl(run_dir / "routes" / "route_b.jsonl"):
        sid = row["sample_id"]
        if sid in routes or row["selected_condition"] != ROUTE_LABEL_TO_CONDITION[row["label"]]:
            raise ValueError(f"Invalid frozen route for {sid}")
        routes[sid] = row
    if not routes or routes.keys() != groups.keys():
        raise ValueError("Frozen routes and benchmark questions must have identical IDs")
    conditions = {f"ratio{p}": f"ratio{p}" for p in GRID}
    conditions.update(ratio0="trackb1024", ratio100="tracka1024", router="router")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=output.parent, delete=False) as dest:
        temporary = Path(dest.name)
        try:
            for condition, name in conditions.items():
                seen = set()
                for row in read_jsonl(run_dir / "raw" / f"{name}_b4000.jsonl"):
                    sid = row["sample_id"]
                    if (sid in seen or sid not in routes or row["condition"] != name
                            or row["nominal_budget_chars"] != 4000 or row["question"] != routes[sid]["question"]
                            or not isinstance(row["is_correct"], bool)):
                        raise ValueError(f"Incompatible frozen answer at {name}/{sid}")
                    seen.add(sid)
                    choice = routes[sid]
                    if condition == "router" and row["condition_spec"]["name"] != choice["selected_condition"]:
                        raise ValueError(f"Frozen router prediction uses a different action for {sid}")
                    record = {key: row[key] for key in ("sample_id", "question", "gold_answer", "task_type",
                                                       "prediction", "is_correct", "context_sha256")}
                    record.update(experiment="audit", history_level="full", history_group=groups[sid],
                                  budget=4000, condition=condition, prediction_sha256=digest(row["prediction"]))
                    if condition == "router":
                        record.update(selected_ratio=int(choice["selected_condition"][5:]),
                                      route=dict(label=choice["label"], parse_success=choice["parse_success"]))
                    print(json.dumps(record, ensure_ascii=False), file=dest)
                if seen != routes.keys():
                    raise ValueError(f"Incomplete frozen answers for {name}")
            dest.flush()
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)


def samples(args, measure):
    if args.experiment == "locomo":
        data = load_locomo_samples(str(args.data))
        if args.ids:
            ids = json.loads(args.ids.read_text())
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate sample IDs")
            by_id = {sample.sample_id: sample for sample in data}
            data = [by_id[sid] for sid in ids]
        else:
            random.Random(args.seed).shuffle(data)
            data = data[:200]
        for sample in data[: args.limit]:
            yield sample, "full", {}
        return
    raw = json.loads(args.data.read_text())
    if args.ids:
        ids = json.loads(args.ids.read_text())
        by_id = {r["question_id"]: r for r in raw}
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sample IDs")
        raw = [by_id[sid] for sid in ids]
    elif args.experiment == "history":
        random.Random(args.seed).shuffle(raw)
        raw = raw[:200]
    groups = source_history_groups(raw) if args.experiment == "audit" else {}
    for item in raw[: args.limit]:
        levels = args.background if args.experiment == "history" else ["full"]
        for level in levels:
            row = item
            metadata = {}
            if args.experiment == "history":
                count = None if level == "full" else int(level)
                if count is not None and count < 0:
                    raise ValueError("Negative background count")
                row = thin_history(item, count, args.seed)
                metadata = {
                    "session_ids": row["haystack_session_ids"],
                    "source_tokens": measure(
                        "\n".join(msg["content"] for session in row["haystack_sessions"] for msg in session)
                    ),
                }
            history, ids = LongMemEvalRunner._normalize_history(row)
            if groups:
                metadata["history_group"] = groups[item["question_id"]]
            yield (
                EvalSample(
                    sample_id=row["question_id"],
                    question=row["question"],
                    gold_answer=str(row["answer"]),
                    history=history,
                    session_ids=ids,
                    task_type=LongMemEvalRunner._normalize_task_type(row),
                ),
                level,
                metadata,
            )


def route(client, question):
    response = client.chat(
        messages=[{"role": "user", "content": ROUTE_B_PROMPT.format(question=question)}],
        max_tokens=8,
        temperature=0,
    )
    label, parsed = parse_route_label(response.content)
    return int(ROUTE_LABEL_TO_CONDITION[label][5:]), {
        "label": label,
        "parse_success": parsed,
        "raw_response": response.content,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }


def cells(experiment, budget):
    if experiment == "views":
        return list(MIXTURES)
    if experiment == "joint":
        return ["split", "matched"]
    if experiment == "history":
        return ["ratio0", "ratio50"]
    if experiment == "audit" and budget != 4000:
        return ["ratio0", "ratio50"]
    return [f"ratio{p}" for p in GRID] + ["router"]


MIXTURES = {
    "chunks": [("chunk", 100)],
    "extract": [("extract", 100)],
    "summary": [("summary", 100)],
    "facts": [("fact", 100)],
    "extract+chunks": [("extract", 50), ("chunk", 50)],
    "summary+chunks+facts": [("summary", 25), ("chunk", 50), ("fact", 25)],
}


def evaluate_sample(experiment, sample, level, meta, budgets, answer, judge, cache, measure):
    memory = make_memory(sample, answer, cache, 256 if experiment == "joint" else 1024)
    alternatives = alternative_views(memory, answer) if experiment == "views" else None
    runner = LongMemEvalRunner("", judge_llm=judge)
    frozen_route = None
    if experiment == "token" and 1024 in budgets:
        budgets = [1024] + [b for b in budgets if b != 1024]
    for budget in budgets:
        target = budget
        for condition in cells(experiment, budget):
            start = perf_counter()
            routing = {}
            cap = budget
            reused_route = condition == "router" and frozen_route is not None
            if condition == "router":
                if frozen_route is None:
                    route_start = perf_counter()
                    ratio, routing = route(answer, sample.question)
                    routing["latency_seconds"] = perf_counter() - route_start
                    frozen_route = ratio, routing
                ratio, routing = frozen_route
            elif condition.startswith("ratio"):
                ratio = int(condition[5:])
            else:
                ratio = 50 if condition == "split" else 0
            if experiment in ("token", "views"):
                views = ranked_views(memory, sample.question, alternatives)
                shares = (
                    MIXTURES[condition]
                    if alternatives is not None
                    else [("summary", ratio), ("chunk", 100 - ratio)]
                )
                context, budget_used = pack(views, shares, cap, measure, character_budget=experiment == "views")
            else:
                if condition == "matched":
                    cap = target
                context = char_context(memory, sample.question, cap, ratio)
                budget_used = realized_chars(context)
                if condition == "split":
                    target = budget_used
            prompt = ANSWER_PROMPT.format(
                question=sample.question, memory_context=context or "(no memory context)"
            )
            response = answer.chat(
                messages=[{"role": "user", "content": prompt}], max_tokens=200, temperature=0
            )
            seconds = perf_counter() - start + (routing["latency_seconds"] if reused_route else 0)
            correct, verdict = runner.judge_answer_with_verdict(
                response.content, sample.gold_answer, sample.task_type, sample.question
            )
            yield dict(
                meta,
                sample_id=sample.sample_id,
                question=sample.question,
                task_type=sample.task_type,
                gold_answer=sample.gold_answer,
                experiment=experiment,
                history_level=level,
                budget=budget,
                condition=condition,
                applied_cap=cap,
                selected_ratio=ratio if experiment != "views" else None,
                route=routing,
                context_sha256=digest(context),
                context_chars=len(context),
                budget_used=budget_used,
                answer_model=answer.model,
                main_judge_model=judge.model,
                prediction=response.content,
                prediction_sha256=digest(response.content),
                is_correct=correct,
                verdict=verdict,
                answer_input_tokens=response.input_tokens,
                answer_output_tokens=response.output_tokens,
                latency_seconds=seconds,
                latency_mode="reused_route_plus_answer_path" if reused_route else "end_to_end",
            )


def rescore(row, judge):
    mode = "question_inclusive" if row.get("task_type") == "abstention" else "questionless"
    prompt = judge_prompt(row, mode)
    result = judge.chat(messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=8)
    return dict(
        row,
        prediction_sha256=digest(row["prediction"]),
        independent_model=judge.model,
        independent_correct=parse_binary_verdict(result.content),
        independent_verdict=result.content,
    )


BUDGETS = {
    "joint": [1000, 2000, 4000, 6000],
    "audit": [1000, 2000, 4000, 6000],
    "token": [256, 512, 1024, 1536],
    "views": [1000, 4000],
    "history": [500, 1000, 2000, 4000],
    "locomo": [4000],
}


def run(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.cache is None:
        args.cache = Path("data/cache") / args.answer_model.replace("/", "_") / "summary_cache"
    measure = len
    if args.experiment in ("token", "history"):
        if not args.tokenizer:
            raise ValueError("Supply the answer model's local tokenizer.json with --tokenizer")
        measure = token_counter(args.tokenizer)
    budgets = args.budgets or BUDGETS[args.experiment]
    if min(budgets) < 1 or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be positive and unique")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(provenance="new reconstruction", started_at=utc_now(), budgets=budgets,
                    data_sha256=file_sha256(args.data))
    if args.tokenizer:
        metadata["tokenizer_sha256"] = sha256(args.tokenizer.read_bytes()).hexdigest()
    answer, judge = LLMClient(args.answer_model), LLMClient(args.judge_model)
    metadata.update(answer_deployment=answer.api_model, judge_deployment=judge.api_model)
    write_json(args.output / "manifest.json", metadata)
    seen = set()
    for sample, level, meta in samples(args, measure):
        key = (sample.sample_id, level)
        if key in seen:
            raise ValueError(f"Duplicate item/level {key}")
        seen.add(key)
        for row in evaluate_sample(
            args.experiment, sample, level, meta, budgets, answer, judge, str(args.cache), measure
        ):
            append_jsonl(args.output / "answers.jsonl", row)
    if not seen:
        raise ValueError("Empty sample selection")
    write_json(args.output / "complete.json", {"items_and_levels": len(seen), "completed_at": utc_now()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("experiment", choices=BUDGETS)
    gen.add_argument("--data", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--limit", type=int)
    gen.add_argument("--ids", type=Path)
    gen.add_argument("--budgets", nargs="+", type=int)
    gen.add_argument("--background", nargs="+", default=["8", "full"])
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--answer-model", default="deepseek-v4-flash")
    gen.add_argument("--judge-model", default="deepseek-v4-flash")
    gen.add_argument("--tokenizer", type=Path)
    gen.add_argument("--cache", type=Path)
    frozen = sub.add_parser("prepare-audit", help="Read the frozen main-sweep answers for Appendix G")
    frozen.add_argument("--data", type=Path, required=True)
    frozen.add_argument("--run-dir", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("rescore")
    score.add_argument("--answers", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--judge-model", default="gpt-4o-mini")
    args = p.parse_args()
    if args.command == "generate":
        run(args)
    elif args.command == "prepare-audit":
        prepare_audit(args.data, args.run_dir, args.output)
    else:
        judge = LLMClient(args.judge_model)
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.answers.name == "answers.jsonl" and (args.answers.parent / "manifest.json").exists():
            if not (args.answers.parent / "complete.json").exists():
                raise ValueError("Generation is incomplete")
        with NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as dest:
            temporary = Path(dest.name)
        try:
            with args.answers.open() as source, temporary.open("w") as dest:
                for line in source:
                    dest.write(json.dumps(rescore(json.loads(line), judge), ensure_ascii=False) + "\n")
            os.link(temporary, args.output)
        finally:
            temporary.unlink()


if __name__ == "__main__":
    main()
