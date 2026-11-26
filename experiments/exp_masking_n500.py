"""Table 13: mask Track A or Track B after writing the full history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp_paper_controls import read_jsonl
from experiments.exp_orthogonality import stored_track_contexts
from src.evaluation.longmemeval_runner import (
    ANSWER_PROMPT,
    LongMemEvalRunner,
    MERGED_SESSION_ID,
)
from src.memory.baselines.strategies import (
    BaseMemoryStrategy,
    DualTrackStrategy,
    DualTrackMaskAStrategy,
    DualTrackMaskBStrategy,
)
from src.models.llm_client import LLMClient


PROGRESS_FLUSH_EVERY = 25
MEM_CTX_PREVIEW_CHARS = 8000          # how much of the injected memory to keep
TRACK_CTX_PREVIEW_CHARS = 8000        # how much of each track context to keep
GOLD_PREVIEW_CHARS = 200
JUDGE_PREVIEW_CHARS = 400
ANSWER_MAX_TOKENS = 200


# ── Strategy factories ────────────────────────────────────────────

def build_strategy(name: str, llm_client: LLMClient, persist_root: str) -> BaseMemoryStrategy:
    """Build a fresh strategy instance for one masking-ablation condition."""
    persist_dir = os.path.join(persist_root, name)
    summary_cache_dir = os.path.join(persist_root, "summary_cache")
    if name == "dual_track":
        return DualTrackStrategy(
            llm_client=llm_client,
            persist_dir=persist_dir,
            summary_cache_dir=summary_cache_dir,
        )
    if name == "dual_track_mask_a":
        return DualTrackMaskAStrategy(
            llm_client=llm_client, persist_dir=persist_dir,
            summary_cache_dir=summary_cache_dir,
        )
    if name == "dual_track_mask_b":
        return DualTrackMaskBStrategy(
            llm_client=llm_client, persist_dir=persist_dir,
            summary_cache_dir=summary_cache_dir,
        )
    raise ValueError(f"unknown condition: {name}")


# ── Helpers ───────────────────────────────────────────────────────

def atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def jsonl_append_line(file_obj, row: dict) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_obj.flush()
    os.fsync(file_obj.fileno())


def scan_done_ids(jsonl_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = row.get("sample_id")
                if sid:
                    done.add(sid)
            except json.JSONDecodeError:
                # truncated last line from a prior crash; ignore
                continue
    return done


def truncate(s: str, n: int) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"...[truncated {len(s) - n} chars]"


# ── Manifest ──────────────────────────────────────────────────────

@dataclass
class ManifestState:
    path: str
    payload: dict

    @classmethod
    def init_or_load(
        cls, output_dir: str, args: argparse.Namespace, conditions: List[str],
    ) -> "ManifestState":
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "manifest.json")
        settings = dict(data_path=args.data_path, budget_chars=args.budget, answer_model=args.model,
                        judge_model=args.judge_model, conditions=conditions, max_samples_cap=args.max_samples,
                        split_sha256=hashlib.sha256(args.split_input.read_bytes()).hexdigest() if args.split_input else None)
        if os.path.exists(path):
            with open(path) as f:
                payload = json.load(f)
            if any(payload.get(key) != value for key, value in settings.items()):
                raise ValueError(f"Masking settings or source predictions differ from {path}")
        else:
            payload = dict(settings, completed_cells={c: {"completed_n": 0, "total_n": None,
                                                          "errors": 0} for c in conditions})
            atomic_write_json(path, payload)
        return cls(path=path, payload=payload)

    def update_cell(self, cond: str, completed_n: int, total_n: int,
                    errors_inc: int = 0) -> None:
        cell = self.payload["completed_cells"][cond]
        cell["completed_n"] = completed_n
        cell["total_n"] = total_n
        cell["errors"] = cell.get("errors", 0) + errors_inc
        atomic_write_json(self.path, self.payload)

# ── Per-sample runner ─────────────────────────────────────────────

def retrieve_context(strategy, runner, sample, budget_chars):
    is_dual_track = (
        hasattr(strategy, "_mem")
        and hasattr(strategy._mem, "get_context_with_attribution")
    )

    # ── 1. setup history ────────────────────────────────────────
    use_merged = len(sample.session_ids) > 1
    touched = set(sample.session_ids)
    if use_merged:
        touched.add(MERGED_SESSION_ID)
    for sid in touched:
        strategy.clear(sid)

    turn = 0
    current_session = sample.session_ids[0] if sample.session_ids else "default"
    for pair in runner._iter_turn_pairs(sample.history):
        sid = pair.get("session_id", current_session)
        write_sid = MERGED_SESSION_ID if (use_merged and is_dual_track) else sid
        touched.add(write_sid)
        if is_dual_track and use_merged:
            strategy.add_turn(write_sid, turn, pair["user"], pair["assistant"],
                              real_session_id=sid)
        else:
            strategy.add_turn(write_sid, turn, pair["user"], pair["assistant"])
        if use_merged and not is_dual_track:
            strategy.add_turn(MERGED_SESSION_ID, turn, pair["user"], pair["assistant"])
        turn += 1

    # ── 2. context (with attribution) ───────────────────────────
    context_session_id = MERGED_SESSION_ID if use_merged else current_session
    track_a_ctx = ""
    track_b_ctx = ""
    if is_dual_track:
        mem_ctx, attribution = strategy._mem.get_context_with_attribution(
            context_session_id, query=sample.question, budget_chars=budget_chars,
        )
        track_a_ctx = attribution.get("track_a_context", "") or ""
        track_b_ctx = attribution.get("track_b_context", "") or ""
    else:
        mem_ctx = strategy.get_context(
            context_session_id, query=sample.question, budget_chars=budget_chars,
        )

    return mem_ctx or "", track_a_ctx, track_b_ctx


def run_one_sample(
    strategy: BaseMemoryStrategy,
    runner: LongMemEvalRunner,
    answer_llm: LLMClient,
    sample,
    budget_chars: int,
    condition: str,
    max_retries: int = 3,
    source=None,
) -> dict:
    """Run a single (strategy, sample) cell and return a JSON-serializable row."""
    if source is None:
        mem_ctx, track_a_ctx, track_b_ctx = retrieve_context(strategy, runner, sample, budget_chars)
    else:
        track_a_ctx, track_b_ctx = stored_track_contexts(source)
        full = "\n\n".join(part for part in (track_a_ctx, track_b_ctx) if part)
        if (source["sample_id"] != sample.sample_id or source["question"] != sample.question
                or source["nominal_budget_chars"] != budget_chars or source["condition"] != "ratio50"
                or source["gold_answer"] != sample.gold_answer or full != source["memory_context"]):
            raise ValueError(f"Invalid stored Split input for {sample.sample_id}")
        if condition == "dual_track":
            return dict(source, condition=condition, answer=source["prediction"], budget_chars=budget_chars,
                        mem_ctx_chars=len(full), track_a_chars=len(track_a_ctx), track_b_chars=len(track_b_ctx))
        if condition == "dual_track_mask_a":
            track_a_ctx = ""
        elif condition == "dual_track_mask_b":
            track_b_ctx = ""
        else:
            raise ValueError(f"Unknown mask {condition}")
        mem_ctx = track_a_ctx or track_b_ctx

    # ── 3. answer LLM with retry ────────────────────────────────
    prompt = ANSWER_PROMPT.format(
        memory_context=mem_ctx if mem_ctx else "(no memory context)",
        question=sample.question,
    )
    prompt_template = "ANSWER_PROMPT"

    ts_ans_start = time.time()
    last_err = None
    resp = None
    retries = 0
    for attempt in range(max_retries):
        try:
            resp = answer_llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=ANSWER_MAX_TOKENS,
            )
            break
        except Exception as e:
            last_err = repr(e)
            retries += 1
            time.sleep(2 ** attempt)
    if resp is None:
        raise RuntimeError(f"Answer call failed for {sample.sample_id}: {last_err}")
    ts_ans_end = time.time()

    # ── 4. judge with retry ─────────────────────────────────────
    ts_judge_start = time.time()
    is_correct = False
    judge_verdict_raw = ""
    for attempt in range(max_retries):
        try:
            is_correct, judge_verdict_raw = runner.judge_answer_with_verdict(
                resp.content,
                sample.gold_answer,
                sample.task_type,
                sample.question,
            )
            break
        except Exception as e:
            last_err = repr(e)
            retries += 1
            time.sleep(2 ** attempt)
    if not judge_verdict_raw:
        raise RuntimeError(f"Judge call failed for {sample.sample_id}: {last_err}")
    ts_judge_end = time.time()

    return {
        "sample_id": sample.sample_id,
        "task_type": sample.task_type,
        "condition": condition,
        "budget_chars": budget_chars,
        "is_correct": bool(is_correct),
        "answer": resp.content,
        "judge_verdict_raw": truncate(judge_verdict_raw, JUDGE_PREVIEW_CHARS),
        "mem_ctx_chars": len(mem_ctx),
        "mem_ctx_preview": truncate(mem_ctx, MEM_CTX_PREVIEW_CHARS),
        "track_a_chars": len(track_a_ctx),
        "track_b_chars": len(track_b_ctx),
        "track_a_preview": truncate(track_a_ctx, TRACK_CTX_PREVIEW_CHARS),
        "track_b_preview": truncate(track_b_ctx, TRACK_CTX_PREVIEW_CHARS),
        "prompt_template": prompt_template,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "thinking_tokens": getattr(resp, "thinking_tokens", 0) or 0,
        "answer_model": getattr(answer_llm, "requested_model", None) or "deepseek-v4-flash",
        "answer_model_resolved": getattr(answer_llm, "model", None),
        "judge_model": getattr(getattr(runner, "judge_llm", None), "model_name", None),
        "judge_model_resolved": getattr(getattr(runner, "judge_llm", None), "model", None),
        "retry_count": retries,
        "last_error": last_err if retries > 0 else None,
    }


# ── Per-condition orchestration ───────────────────────────────────

def run_condition(
    condition: str,
    samples: list,
    runner: LongMemEvalRunner,
    answer_llm: LLMClient,
    output_dir: str,
    budget: int,
    persist_root: str,
    manifest: ManifestState,
    max_samples: Optional[int],
    sources=None,
) -> None:
    out_path = os.path.join(output_dir, f"per_sample_{condition}_b{budget}.jsonl")
    done_ids = scan_done_ids(out_path)
    target = samples
    if max_samples is not None:
        target = samples[:max_samples]
    pending = [s for s in target if s.sample_id not in done_ids]
    total_n = len(target)
    print(f"\n=== {condition} (b={budget}) ===  done={len(done_ids)}/{total_n}  pending={len(pending)}")

    if not pending:
        manifest.update_cell(condition, completed_n=len(done_ids), total_n=total_n)
        return

    strategy = build_strategy(condition, llm_client=answer_llm, persist_root=persist_root) if sources is None else None
    completed = len(done_ids)
    with open(out_path, "a") as f:
        for i, sample in enumerate(pending, start=1):
            row = run_one_sample(
                strategy=strategy,
                runner=runner,
                answer_llm=answer_llm,
                sample=sample,
                budget_chars=budget,
                condition=condition,
                source=sources[sample.sample_id] if sources is not None else None,
            )
            jsonl_append_line(f, row)
            completed += 1

            if i % PROGRESS_FLUSH_EVERY == 0 or i == len(pending):
                print(f"  [{condition}] {completed}/{total_n}  "
                      f"sample={sample.sample_id} ({sample.task_type})  "
                      f"acc={row.get('is_correct')}")
                manifest.update_cell(condition, completed_n=completed,
                                     total_n=total_n, errors_inc=0)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_path", default="data/longmemeval/longmemeval_s.json")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--judge_model", default="deepseek-v4-flash")
    parser.add_argument("--split-input", type=Path, help="Stored ratio50 outputs for the same budget and question IDs")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--conditions",
                        default="dual_track,dual_track_mask_a,dual_track_mask_b",
                        help="comma-separated condition names")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional sample limit; default uses all samples")
    parser.add_argument("--persist_root", default=None,
                        help="Where to put DualTrackMemory persist dirs (default: <output_dir>/persist)")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    output_dir = args.output_dir
    persist_root = args.persist_root or os.path.join(
        "data/cache", args.model.replace("/", "_")
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(persist_root, exist_ok=True)

    manifest = ManifestState.init_or_load(output_dir, args, conditions)

    # Load samples
    judge_llm = LLMClient(args.judge_model)
    runner = LongMemEvalRunner(args.data_path, judge_llm=judge_llm)
    if runner._data is None:
        runner._data = runner._load_data()
    samples = runner._data
    print(f"Loaded {len(samples)} samples from {args.data_path}")

    sources = None
    if args.split_input:
        records = read_jsonl(args.split_input)
        sources = {row["sample_id"]: row for row in records}
        target_ids = {sample.sample_id for sample in samples[:args.max_samples]}
        if len(sources) != len(records) or not target_ids.issubset(sources):
            raise ValueError("Missing or duplicate Split predictions for masking")

    # Build LLM clients
    answer_llm = LLMClient(args.model)

    # Run each remaining condition
    for cond in conditions:
        run_condition(
            condition=cond,
            samples=samples,
            runner=runner,
            answer_llm=answer_llm,
            output_dir=output_dir,
            budget=args.budget,
            persist_root=persist_root,
            manifest=manifest,
            max_samples=args.max_samples,
            sources=sources,
        )

    print("\nDone.")
    print(f"manifest:        {manifest.path}")
    print(f"per-sample dir:  {output_dir}")


if __name__ == "__main__":
    main()
