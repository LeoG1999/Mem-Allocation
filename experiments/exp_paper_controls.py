#!/usr/bin/env python3
"""Experiments for Tables 2, 6, 9--11, 14, and 16--18."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.longmemeval_runner import JUDGE_PROMPT, JUDGE_PROMPT_ABSTENTION, LongMemEvalRunner
from src.memory.baselines.strategies import BM25ChunkTrackBStrategy
from src.memory.track_b_rag import payload_chars
from src.memory.dual_track import DualTrackMemory
from src.models.llm_client import LLMClient


ROUTE_LABEL_TO_CONDITION = {
    "EVIDENCE": "ratio25",
    "STATE": "ratio75",
    "MIXED": "ratio50",
}

ROUTE_B_PROMPT = """You will see one question addressed to a long-term conversational assistant that has memory of past sessions with the user. Decide which kind of memory context the question primarily needs. Choose exactly one label:

1. EVIDENCE: the answer depends on locating specific details stated in past conversation turns, such as numbers, dates, names, exact wordings, or isolated events.
2. STATE: the answer depends on the user's current preferences, habits, plans, or the latest state of an evolving situation, where earlier information may have been superseded.
3. MIXED: the question needs both turn-level details and session-level state, or neither clearly dominates.

Question: {question}

Reply with only one word: EVIDENCE, STATE, or MIXED."""

JUDGE_QUESTION_INCLUSIVE = JUDGE_PROMPT.replace("Reference answer:", "Question: {question}\nReference answer:", 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_no}") from exc
    return rows


def load_sample_id_filter(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("sample_ids") or payload.get("plan", {}).get("sample_ids")
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON list or contain sample_ids")
    return {str(value) for value in payload}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def chunk_protocol(chunk_chars: int) -> dict:
    if chunk_chars <= 0 or chunk_chars % 4:
        raise ValueError("chunk chars must be a positive multiple of four")
    return {
        "chunk_size_chars": chunk_chars,
        "chunk_overlap_chars": chunk_chars // 8,
        "dual_track_chunk_size": chunk_chars // 4,
        "dual_track_chunk_overlap": chunk_chars // 32,
    }


def condition_spec(name: str) -> dict:
    presets = {
        "split256": ("split", 0.50, 256),
        "split512": ("split", 0.50, 512),
        "split1024": ("split", 0.50, 1024),
        "trackb256": ("concentration", 0.00, 256),
        "trackb512": ("concentration", 0.00, 512),
        "trackb1024": ("concentration", 0.00, 1024),
        "ratio25": ("split", 0.25, 1024),
        "ratio33": ("split", 0.33, 1024),
        "ratio50": ("split", 0.50, 1024),
        "ratio67": ("split", 0.67, 1024),
        "ratio75": ("split", 0.75, 1024),
        "tracka1024": ("split", 1.00, 1024),
        "router": ("split", 0.50, 1024),
        "split_extended": ("split", 0.50, 1024),
    }
    if name not in presets:
        raise ValueError(f"Unknown condition {name!r}; choose from {sorted(presets)}")
    kind, ratio, chunk_chars = presets[name]
    return {
        "name": name,
        "kind": kind,
        "track_a_ratio": ratio,
        "track_b_ratio": 1.0 - ratio,
        "track_b_top_k": None,
        **chunk_protocol(chunk_chars),
    }


class AllocationDualStrategy:
    """Static allocation wrapper with explicit chunk and cache parameters."""

    def __init__(
        self,
        *,
        name: str,
        llm_client: LLMClient,
        spec: dict,
        summary_cache_dir: str,
        use_llm_summary: bool,
        packing_mode: str,
    ) -> None:
        self.name = name
        self._mem = DualTrackMemory(
            llm_client=llm_client,
            use_llm_memo=use_llm_summary,
            use_router=False,
            track_a_ratio=spec["track_a_ratio"],
            top_k=spec.get("track_b_top_k"),
            chunk_size=spec["dual_track_chunk_size"],
            chunk_overlap=spec["dual_track_chunk_overlap"],
            summary_cache_dir=summary_cache_dir,
            track_b_packing_mode=packing_mode,
        )

    def add_turn(self, session_id, turn, user_content, assistant_content, real_session_id=None):
        self._mem.add_turn(
            session_id,
            turn,
            user_content,
            assistant_content,
            real_session_id=real_session_id,
        )

    def get_context(self, session_id, query="", budget_chars=None):
        return self._mem.get_context(session_id, query=query, budget_chars=budget_chars)

    def clear(self, session_id):
        self._mem.clear(session_id)


def build_strategy(spec: dict, llm: LLMClient, args):
    if spec["kind"] == "concentration":
        strategy = BM25ChunkTrackBStrategy(
            chunk_size_chars=spec["chunk_size_chars"],
            chunk_overlap_chars=spec["chunk_overlap_chars"],
            packing_mode=args.packing_mode,
            budget_includes_wrapper=False,
            top_k=spec.get("track_b_top_k"),
        )
        strategy.name = spec["name"]
        return strategy
    return AllocationDualStrategy(
        name=spec["name"],
        llm_client=llm,
        spec=spec,
        summary_cache_dir=args.summary_cache_dir,
        use_llm_summary=True,
        packing_mode=args.packing_mode,
    )


def select_samples(runner: LongMemEvalRunner, max_samples: Optional[int], sample_ids: list[str]) -> list:
    if runner._data is None:
        runner._data = runner._load_data()
    data = runner._data
    if sample_ids:
        wanted = set(sample_ids)
        data = [s for s in data if s.sample_id in wanted]
        missing = wanted - {s.sample_id for s in data}
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)[:5]}")
    if max_samples is not None:
        data = data[:max_samples]
    return list(data)


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", default="data/longmemeval/longmemeval_s.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")


def reuse_main(args) -> None:
    """Expose the original sweep's policy endpoints to the control workflows."""
    main_runs = json.loads(Path(args.main).read_text())["raw_runs"]
    if len(main_runs) != 1:
        raise ValueError("Select a single main sweep for the controls")
    track = json.loads(Path(args.trackb).read_text())["bm25_chunk_trackb"]
    questions = {s.sample_id: s.question for s in LongMemEvalRunner(args.data_path)._load_data()}
    raw = Path(args.output_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, cell in (("ratio50", main_runs[0]["dual_track"]),
                       ("tracka1024", main_runs[0]["session_summary"]), ("trackb1024", track)):
        spec = condition_spec(name)
        if len(cell["budgets"]) != len(cell["sample_correctness"]):
            raise ValueError(f"Incomplete budget sweep for {name}")
        for budget, rows in zip(cell["budgets"], cell["sample_correctness"]):
            ids = [row["sample_id"] for row in rows]
            if len(ids) != len(set(ids)) or set(ids) != questions.keys():
                raise ValueError(f"Question IDs differ for {name} at {budget}")
            destination = raw / f"{name}_b{budget}.jsonl"
            with NamedTemporaryFile("w", dir=raw, delete=False) as out:
                temporary = Path(out.name)
                try:
                    for row in rows:
                        context = row["memory_context"]
                        record = dict(row, question=questions[row["sample_id"]], condition=name,
                                      condition_spec=spec, nominal_budget_chars=budget,
                                      context_sha256=sha256_text(context), main_judge_model=args.judge_model)
                        print(json.dumps(record, ensure_ascii=False), file=out)
                    out.flush()
                    try:
                        os.link(temporary, destination)
                    except FileExistsError:
                        if destination.read_bytes() != temporary.read_bytes():
                            raise ValueError(f"Existing control input differs from the main sweep: {destination}")
                finally:
                    temporary.unlink(missing_ok=True)


def generate(args) -> None:
    specs = [condition_spec(name) for name in args.conditions]
    output_dir = Path(args.output_dir)
    manifest = {
        "schema_version": 1,
        "command": "generate",
        "created_at": utc_now(),
        "data_path": args.data_path,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "judge_mode": args.judge_mode,
        "budgets": args.budgets,
        "conditions": specs,
        "packing_mode": args.packing_mode,
        "store_context": True,
        "summary_cache_dir": args.summary_cache_dir,
        "dry_run": args.dry_run,
    }
    write_json(output_dir / "generate_manifest.json", manifest)

    runner = LongMemEvalRunner(args.data_path)
    samples = select_samples(runner, args.max_samples, args.sample_ids)
    routes = {}
    if "router" in args.conditions:
        routes = {row["sample_id"]: row for row in read_jsonl(Path(args.routes))}
        for sample in samples:
            record = routes.get(sample.sample_id)
            if (not record or record["question"] != sample.question
                    or record["selected_condition"] != ROUTE_LABEL_TO_CONDITION[record["label"]]):
                raise ValueError(f"Missing or incompatible frozen route for {sample.sample_id}")
    print(f"generate: {len(samples)} samples × {len(specs)} conditions × {len(args.budgets)} budgets")
    if args.dry_run:
        print(f"DRY RUN: wrote {output_dir / 'generate_manifest.json'}")
        return

    answer_llm = LLMClient(args.answer_model)
    judge_llm = LLMClient(args.judge_model) if args.judge_mode == "hybrid" else None
    runner.judge_llm = judge_llm

    for spec in specs:
        for budget in args.budgets:
            path = output_dir / "raw" / f"{spec['name']}_b{budget}.jsonl"
            if args.force and path.exists():
                raise ValueError(f"--force does not overwrite append-only artifacts; move {path} first")
            done = {row["sample_id"] for row in read_jsonl(path)}
            strategy = build_strategy(spec, answer_llm, args)
            print(f"[{spec['name']} b={budget}] resume={len(done)} target={len(samples)}")
            for index, sample in enumerate(samples, 1):
                if sample.sample_id in done:
                    continue
                active_spec = spec
                if spec["name"] == "router":
                    active_spec = condition_spec(routes[sample.sample_id]["selected_condition"])
                    strategy._mem.track_a_ratio = active_spec["track_a_ratio"]
                report = runner.evaluate(
                    strategy=strategy,
                    answer_llm=answer_llm,
                    sample_ids=[sample.sample_id],
                    max_samples=1,
                    judge_mode=args.judge_mode,
                    budget_chars=budget,
                    store_context=True,
                    verbose=False,
                )
                result = asdict(report.samples[0])
                context = result.get("memory_context") or ""
                row = {
                    **result,
                    "question": sample.question,
                    "condition": spec["name"],
                    "condition_spec": active_spec,
                    "nominal_budget_chars": budget,
                    "context_sha256": sha256_text(context),
                    "answer_model": answer_llm.model,
                    "main_judge_model": judge_llm.model if judge_llm else None,
                    "main_judge_mode": args.judge_mode,
                    "packing_mode": args.packing_mode,
                    "completed_at": utc_now(),
                }
                append_jsonl(path, row)
                print(
                    f"  {index}/{len(samples)} {sample.sample_id} "
                    f"correct={int(row['is_correct'])} ctx={row['memory_context_len']}",
                    flush=True,
                )


def parse_route_label(text: str) -> tuple[str, bool]:
    # Paper protocol: parse only the first alphabetic token.
    match = re.search(r"[A-Za-z]+", text or "")
    if match:
        label = match.group(0).upper()
        if label in ROUTE_LABEL_TO_CONDITION:
            return label, True
    return "MIXED", False


def route(args) -> None:
    output = Path(args.output)
    runner = LongMemEvalRunner(args.data_path)
    samples = select_samples(runner, args.max_samples, args.sample_ids)
    manifest = {
        "schema_version": 1,
        "command": "route",
        "created_at": utc_now(),
        "data_path": args.data_path,
        "model": args.model,
        "n_samples": len(samples),
        "prompt": ROUTE_B_PROMPT,
        "prompt_sha256": sha256_text(ROUTE_B_PROMPT),
        "mapping": ROUTE_LABEL_TO_CONDITION,
        "question_only": True,
        "dry_run": args.dry_run,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(f"route: {len(samples)} question-only classifications")
    if args.dry_run:
        return
    done = {row["sample_id"] for row in read_jsonl(output)}
    llm = LLMClient(args.model)
    for index, sample in enumerate(samples, 1):
        if sample.sample_id in done:
            continue
        response = llm.chat(
            messages=[{"role": "user", "content": ROUTE_B_PROMPT.format(question=sample.question)}],
            temperature=0.0,
            max_tokens=8,
        )
        label, parsed = parse_route_label(response.content)
        append_jsonl(output, {
            "sample_id": sample.sample_id,
            "task_type": sample.task_type,
            "question": sample.question,
            "label": label,
            "selected_condition": ROUTE_LABEL_TO_CONDITION[label],
            "parse_success": parsed,
            "raw_response": response.content,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "model": llm.model,
            "prompt_sha256": manifest["prompt_sha256"],
        })
        print(f"  {index}/{len(samples)} {sample.sample_id} -> {label} parsed={parsed}", flush=True)


def judge_prompt(row: dict, mode: str) -> str:
    abstention = row.get("task_type") == "abstention"
    if abstention:
        template = JUDGE_PROMPT_ABSTENTION
    elif mode == "question_inclusive":
        template = JUDGE_QUESTION_INCLUSIVE
    else:
        template = JUDGE_PROMPT
    return template.format(
        question=row.get("question", ""),
        gold=row.get("gold_answer", ""),
        prediction=row.get("prediction", ""),
    )


def parse_yes_no(text: str) -> tuple[bool, bool]:
    match = re.search(r"\b(YES|NO)\b", (text or "").upper())
    if not match:
        return False, False
    return match.group(1) == "YES", True


def rejudge(args) -> None:
    model = None if args.dry_run else LLMClient(args.judge_model)
    output_dir = Path(args.output_dir)
    id_filter = load_sample_id_filter(args.sample_ids_file)
    for input_name in args.inputs:
        source = Path(input_name)
        rows = read_jsonl(source)
        if id_filter is not None:
            rows = [row for row in rows if str(row["sample_id"]) in id_filter]
        if args.limit is not None:
            rows = rows[:args.limit]
        tag = f"_{args.tag}" if args.tag else ""
        output = output_dir / f"{source.stem}_{args.prompt_mode}_{args.judge_model.replace('/', '_')}{tag}.jsonl"
        manifest = {
            "schema_version": 1,
            "command": "rejudge",
            "source": str(source),
            "source_sha256": sha256_text(source.read_text(encoding="utf-8")),
            "judge_model": args.judge_model,
            "prompt_mode": args.prompt_mode,
            "n_rows": len(rows),
            "sample_ids_file": args.sample_ids_file,
            "sample_ids_sha256": sha256_text("\n".join(sorted(r["sample_id"] for r in rows))),
            "prompt_sha256": sha256_text(
                JUDGE_QUESTION_INCLUSIVE if args.prompt_mode == "question_inclusive" else JUDGE_PROMPT
            ),
            "abstention_prompt_sha256": sha256_text(JUDGE_PROMPT_ABSTENTION),
            "all_predictions_judged_by_llm": True,
            "dry_run": args.dry_run,
        }
        manifest_path = output.with_suffix(".manifest.json")
        existing = read_jsonl(output)
        done = {r["sample_id"] for r in existing}
        expected = {r["sample_id"] for r in rows}
        if len(expected) != len(rows) or len(done) != len(existing) or not done.issubset(expected):
            raise ValueError(f"Inconsistent question IDs for {output}")
        if existing:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("source_sha256", "sample_ids_sha256", "judge_model", "prompt_mode",
                        "prompt_sha256", "abstention_prompt_sha256"):
                if previous.get(key) != manifest[key]:
                    raise ValueError(f"Existing judgments use a different {key}: {output}")
        elif not args.dry_run:
            write_json(manifest_path, manifest)
        print(f"rejudge: {source} -> {output} ({len(rows)} decisions)")
        if args.dry_run:
            continue
        for index, row in enumerate(rows, 1):
            if row["sample_id"] in done:
                continue
            response = model.chat(
                messages=[{"role": "user", "content": judge_prompt(row, args.prompt_mode)}],
                temperature=0.0,
                max_tokens=8,
            )
            correct, parsed = parse_yes_no(response.content)
            if not parsed:
                raise ValueError(
                    f"Judge verdict is not YES/NO for {row['sample_id']}: {response.content!r}"
                )
            append_jsonl(output, {
                "sample_id": row["sample_id"],
                "task_type": row.get("task_type"),
                "condition": row.get("condition"),
                "nominal_budget_chars": row.get("nominal_budget_chars"),
                "source_context_sha256": row.get("context_sha256"),
                "source_prediction_sha256": sha256_text(row.get("prediction", "")),
                "source_main_correct": row.get("is_correct"),
                "is_correct": correct,
                "parse_success": parsed,
                "raw_verdict": response.content,
                "judge_model": model.model,
                "prompt_mode": args.prompt_mode,
                "prompt_sha256": manifest["abstention_prompt_sha256" if row.get("task_type") == "abstention" else "prompt_sha256"],
            })
            print(f"  {index}/{len(rows)} {row['sample_id']} -> {response.content.strip()}", flush=True)


def length_match(args) -> None:
    output_dir = Path(args.output_dir)
    base_runner = LongMemEvalRunner(args.data_path)
    all_samples = select_samples(base_runner, None, [])
    sample_index = {s.sample_id: s for s in all_samples}
    if args.dry_run:
        print(f"length-match dry run: {len(args.split_inputs)} source cells")
        return
    answer_llm = LLMClient(args.answer_model)
    judge_llm = LLMClient(args.judge_model) if args.judge_mode == "hybrid" else None
    base_runner.judge_llm = judge_llm

    for input_name in args.split_inputs:
        source = Path(input_name)
        source_rows = read_jsonl(source)
        if args.limit is not None:
            source_rows = source_rows[:args.limit]
        if not source_rows:
            continue
        chunk_chars = int(source_rows[0]["condition_spec"]["chunk_size_chars"])
        spec = {
            "name": f"trackb{chunk_chars}_length_matched",
            "kind": "concentration",
            "track_a_ratio": 0.0,
            "track_b_ratio": 1.0,
            **chunk_protocol(chunk_chars),
        }
        strategy = build_strategy(spec, answer_llm, args)
        output = output_dir / f"{source.stem}_length_matched.jsonl"
        done = {row["sample_id"] for row in read_jsonl(output)}
        write_json(output.with_suffix(".manifest.json"), {
            "schema_version": 1,
            "command": "length-match",
            "source": str(source),
            "matching_unit": "budgeted_memory_characters_excluding_fixed_track_b_delimiters",
            "packing_mode": args.packing_mode,
            "whole_units_only": args.packing_mode == "whole_unit_skip",
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
        })
        for index, split_row in enumerate(source_rows, 1):
            sid = split_row["sample_id"]
            if sid in done:
                continue
            if sid not in sample_index:
                raise ValueError(f"Sample {sid} from {source} not present in {args.data_path}")
            attr = split_row.get("attribution") or {}
            a = attr.get("track_a_packing") or {}
            b = attr.get("track_b_packing") or {}
            a_budgeted = int(a.get("serialized_chars", len(attr.get("track_a_context") or "")))
            b_budgeted = payload_chars(b, attr.get("track_b_context") or "")
            separator = 2 if a_budgeted and b_budgeted else 0
            target = a_budgeted + b_budgeted + separator
            report = base_runner.evaluate(
                strategy=strategy,
                answer_llm=answer_llm,
                sample_ids=[sid],
                max_samples=1,
                judge_mode=args.judge_mode,
                budget_chars=target,
                store_context=True,
            )
            result = asdict(report.samples[0])
            achieved = payload_chars(result.get("packing_stats") or {}, result.get("memory_context") or "")
            append_jsonl(output, {
                **result,
                "question": sample_index[sid].question,
                "condition": spec["name"],
                "condition_spec": spec,
                "nominal_budget_chars": split_row["nominal_budget_chars"],
                "split_source_condition": split_row["condition"],
                "split_source_context_sha256": split_row["context_sha256"],
                "split_target_context_chars": target,
                "trackb_achieved_context_chars": achieved,
                "absolute_mismatch_chars": abs(target - achieved),
                "relative_mismatch": abs(target - achieved) / max(target, 1),
                "context_sha256": sha256_text(result.get("memory_context") or ""),
                "packing_mode": args.packing_mode,
            })
            print(f"  {index}/{len(source_rows)} {sid} target={target} achieved={achieved}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reuse-main", help="Use saved predictions and contexts for the repeated main cells")
    p.add_argument("--main", required=True)
    p.add_argument("--trackb", required=True)
    p.add_argument("--data-path", default="data/longmemeval/longmemeval_s.json")
    p.add_argument("--output-dir", default="data/results/paper_controls")
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.set_defaults(func=reuse_main)

    p = sub.add_parser("generate", help="Run static/chunk controls with full per-item artifacts")
    add_common_data_args(p)
    p.add_argument("--output-dir", default="data/results/paper_controls")
    p.add_argument("--answer-model", default="deepseek-v4-flash")
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.add_argument("--judge-mode", choices=["hybrid", "string", "exact"], default="hybrid")
    p.add_argument("--budgets", type=int, nargs="+", default=[1000, 2000, 4000, 6000])
    p.add_argument("--routes", default="data/results/paper_controls/routes/route_b.jsonl")
    p.add_argument("--conditions", nargs="+", default=[
        "split256", "trackb256", "split512", "trackb512",
        "ratio25", "ratio33", "ratio50", "ratio67", "ratio75",
        "trackb1024", "tracka1024",
    ])
    p.add_argument("--packing-mode", choices=["whole_unit_skip", "truncate_last"], default="whole_unit_skip")
    p.add_argument("--summary-cache-dir", default="data/cache/deepseek-v4-flash/summary_cache")
    p.set_defaults(func=generate)

    p = sub.add_parser("route", help="Generate frozen question-only Route-B labels")
    add_common_data_args(p)
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--output", default="data/results/paper_controls/routes/route_b.jsonl")
    p.set_defaults(func=route)

    p = sub.add_parser("rejudge", help="Judge stored predictions without regenerating answers")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-dir", default="data/results/paper_controls/rejudge")
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.add_argument("--prompt-mode", choices=["questionless", "question_inclusive"], required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-ids-file", default=None, help="JSON list or audit_plan.json containing sample_ids")
    p.add_argument("--tag", default=None, help="Output suffix for a named audit subset")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=rejudge)

    p = sub.add_parser("length-match", help="Run Track-B at each stored Split context length")
    p.add_argument("--split-inputs", nargs="+", required=True)
    p.add_argument("--data-path", default="data/longmemeval/longmemeval_s.json")
    p.add_argument("--output-dir", default="data/results/paper_controls/length_matched")
    p.add_argument("--answer-model", default="deepseek-v4-flash")
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.add_argument("--judge-mode", choices=["hybrid", "string", "exact"], default="hybrid")
    p.add_argument("--packing-mode", choices=["whole_unit_skip", "truncate_last"], default="whole_unit_skip")
    p.add_argument("--summary-cache-dir", default="data/cache/deepseek-v4-flash/summary_cache")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=length_match)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
