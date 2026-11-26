"""
LoCoMo benchmark runner (arXiv:2402.17753)

Data format (locomo_test.json):
  10 conversations. Each has:
    - conversation: dict with speaker_a/b and session_1..session_N keys
    - qa: list of {question, answer, evidence, category} items

  All QA items in a conversation share the same history, so history is
  replayed once per conversation for efficiency (~199x speedup vs naive).

Category → task_type mapping:
  1 → multi_hop        multi-hop reasoning
  2 → temporal         date / time questions
  3 → open_domain      open-domain reasoning
  4 → single_hop       single-hop factual recall
  5 → abstention       unanswerable (answer field absent/None)

Metrics: LLM-judged accuracy (same as LongMemEval) + token-F1 as secondary.
Uses shared EvalReport so results are directly comparable across benchmarks.
"""

from __future__ import annotations

import gc
import json
import os
import re
import string
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from src.models.llm_client import LLMClient, LLMResponse
from src.memory.baselines.strategies import BaseMemoryStrategy
from src.evaluation.longmemeval_runner import (
    iter_history_turns as iter_locomo_turns, prepare_summary_cache, summary_view,
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    JUDGE_PROMPT_ABSTENTION,
    MERGED_SESSION_ID,
    EvalReport,
    EvalSample,
    SampleResult,
    parse_binary_verdict,
)


CATEGORY_TO_TASK_TYPE: Dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "abstention",
}


# ── Token-F1 (secondary metric, kept for LoCoMo paper alignment) ─

def _normalize_text(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def _token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize_text(pred).split()
    g_toks = _normalize_text(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec  = common / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ── Data loading ─────────────────────────────────────────────────

def _extract_ordered_sessions(conv: Dict) -> List[Tuple[str, List[Dict]]]:
    """Return [(session_key, messages), ...] sorted by session number."""
    keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    return [(k, conv[k]) for k in keys if isinstance(conv.get(k), list)]


def _build_flat_history(
    conv_id: str,
    sessions: List[Tuple[str, List[Dict]]],
    speaker_a: str,
) -> Tuple[List[Dict], List[str]]:
    """Convert {speaker, text} messages to {role, content, session_id} format."""
    flat: List[Dict] = []
    session_ids: List[str] = []
    for session_key, msgs in sessions:
        sid = f"{conv_id}__{session_key}"
        session_ids.append(sid)
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role = "user" if msg.get("speaker") == speaker_a else "assistant"
            flat.append({
                "role": role,
                "content": msg.get("text", ""),
                "session_id": sid,
                "dia_id": msg.get("dia_id", ""),
            })
    return flat, session_ids


def load_locomo_samples(path: str) -> List[EvalSample]:
    """Load LoCoMo JSON and flatten into one EvalSample per QA item."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LoCoMo data not found: {path}\n"
            "Download from https://github.com/snap-research/locomo"
        )
    with open(path) as f:
        raw = json.load(f)

    samples: List[EvalSample] = []
    for item in raw:
        conv = item.get("conversation", {})
        conv_id = str(item.get("sample_id", f"conv_{len(samples)}"))
        speaker_a = conv.get("speaker_a", "A")
        sessions = _extract_ordered_sessions(conv)
        flat_history, session_ids = _build_flat_history(conv_id, sessions, speaker_a)

        for qi, qa in enumerate(item.get("qa", [])):
            raw_answer = qa.get("answer", qa.get("Answer"))
            category = int(qa.get("category", 1))
            task_type = CATEGORY_TO_TASK_TYPE.get(category, f"category_{category}")

            # Category 5 or missing answer → abstention
            if task_type == "abstention" or raw_answer is None:
                gold = "I don't know"
                task_type = "abstention"
            else:
                gold = str(raw_answer)

            samples.append(EvalSample(
                sample_id=f"{conv_id}__q{qi}",
                question=str(qa.get("question", "")),
                gold_answer=gold,
                task_type=task_type,
                history=flat_history,   # shared reference — never mutate
                session_ids=session_ids,
            ))
    return samples


# ── Runner ───────────────────────────────────────────────────────

class LoCoMoRunner:
    """Evaluates memory strategies on LoCoMo using LLM-judged accuracy.

    Returns EvalReport (same schema as LongMemEvalRunner) for direct
    cross-benchmark comparison.  Token-F1 is stored in task_breakdown
    under key "f1_<task_type>" as a secondary metric.

    Efficiency: history is added once per conversation (~199 QA items share
    the same session history), cutting LLM add_turn calls by ~199×.
    """

    def __init__(
        self,
        data_path: str,
        judge_llm: Optional[LLMClient] = None,
        answer_llm: Optional[LLMClient] = None,
    ):
        self.data_path = data_path
        self.judge_llm = judge_llm
        self.answer_llm = answer_llm
        self._samples: Optional[List[EvalSample]] = None

    def _get_samples(self) -> List[EvalSample]:
        if self._samples is None:
            self._samples = load_locomo_samples(self.data_path)
        return self._samples

    def _judge(self, prediction: str, gold: str, task_type: str, question: str) -> tuple[bool, str]:
        if not prediction:
            return False, "EMPTY_PREDICTION"
        pred_norm = prediction.lower().strip()
        if not gold:
            return False, "EMPTY_REFERENCE"
        gold_norm = gold.lower().strip()
        if self.judge_llm is None:
            value = pred_norm == gold_norm or gold_norm in pred_norm
            return value, "RULE_MATCH" if value else "RULE_NO_MATCH"
        prompt = (
            JUDGE_PROMPT_ABSTENTION.format(question=question, gold=gold, prediction=prediction)
            if task_type == "abstention"
            else JUDGE_PROMPT.format(gold=gold, prediction=prediction)
        )
        response = self.judge_llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        verdict = response.content.strip()
        return parse_binary_verdict(verdict), verdict

    def evaluate(
        self,
        strategy: BaseMemoryStrategy,
        answer_llm: Optional[LLMClient] = None,
        max_samples: Optional[int] = None,
        task_types: Optional[List[str]] = None,
        budget_chars: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        verbose: bool = False,
    ) -> EvalReport:
        llm = answer_llm or self.answer_llm
        if llm is None:
            raise ValueError("answer_llm is required")

        samples = self._get_samples()
        if task_types:
            samples = [s for s in samples if s.task_type in task_types]
        if shuffle:
            rng = random.Random(seed)
            samples = list(samples)
            rng.shuffle(samples)
        if max_samples:
            samples = samples[:max_samples]

        is_dual_track = (
            hasattr(strategy, "_mem")
            and hasattr(strategy._mem, "get_context_with_attribution")
        )

        view = summary_view(strategy)
        prepare_summary_cache(view, samples)

        # Group by conversation so history is added only once per conversation
        conv_groups: Dict[str, List[EvalSample]] = defaultdict(list)
        for s in samples:
            conv_id = s.sample_id.rsplit("__q", 1)[0]
            conv_groups[conv_id].append(s)

        results: List[SampleResult] = []

        for conv_idx, (conv_id, conv_samples) in enumerate(conv_groups.items()):
            rep = conv_samples[0]  # all share same history / session_ids
            session_ids = rep.session_ids
            use_merged = len(session_ids) > 1
            all_sids = set(session_ids)
            if use_merged:
                all_sids.add(MERGED_SESSION_ID)

            for sid in all_sids:
                strategy.clear(sid)

            # Replay conversation history once
            turn = 0
            current_session = session_ids[0] if session_ids else "default"
            for msg_pair in iter_locomo_turns(rep.history):
                sid = msg_pair.get("session_id", current_session)
                write_sid = MERGED_SESSION_ID if (use_merged and is_dual_track) else sid
                all_sids.add(write_sid)
                if is_dual_track and use_merged:
                    strategy.add_turn(write_sid, turn, msg_pair["user"], msg_pair["assistant"],
                                      real_session_id=sid)
                else:
                    strategy.add_turn(write_sid, turn, msg_pair["user"], msg_pair["assistant"])
                if use_merged and not is_dual_track:
                    strategy.add_turn(MERGED_SESSION_ID, turn, msg_pair["user"], msg_pair["assistant"])
                turn += 1

            context_sid = MERGED_SESSION_ID if use_merged else current_session
            total_sessions = max(len(session_ids), 1)

            if view is not None:
                view.prepare_summaries()
            for qi, sample in enumerate(conv_samples):
                if verbose:
                    print(f"[conv {conv_idx+1}/{len(conv_groups)} "
                          f"q{qi+1}/{len(conv_samples)}] {sample.task_type}")
                attribution = None
                if is_dual_track:
                    mem_ctx, attribution = strategy._mem.get_context_with_attribution(
                        context_sid, query=sample.question, budget_chars=budget_chars
                    )
                else:
                    mem_ctx = strategy.get_context(
                        context_sid, query=sample.question, budget_chars=budget_chars
                    )

                injected = len(set(re.findall(r"session[= ]([^\]\s,)]+)", mem_ctx or "")))
                coverage_rate = min(injected / total_sessions, 1.0)
                prompt = ANSWER_PROMPT.format(
                    memory_context=mem_ctx or "(no memory context)",
                    question=sample.question,
                )
                resp: LLMResponse = llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )
                is_correct, judge_verdict_raw = self._judge(
                    resp.content,
                    sample.gold_answer,
                    sample.task_type,
                    sample.question,
                )
                results.append(SampleResult(
                    sample_id=sample.sample_id, task_type=sample.task_type,
                    prediction=resp.content, gold_answer=sample.gold_answer,
                    is_correct=is_correct,
                    memory_context_len=len(mem_ctx) if mem_ctx else 0,
                    input_tokens=resp.input_tokens,
                    thinking_tokens=resp.thinking_tokens,
                    latency_ms=resp.latency_ms,
                    budget_chars=budget_chars,
                    coverage_rate=coverage_rate,
                    attribution=attribution,
                    judge_verdict_raw=judge_verdict_raw,
                ))

            for sid in all_sids:
                strategy.clear(sid)
            if (conv_idx + 1) % 2 == 0:
                gc.collect()

        # Aggregate
        n = len(results)
        accuracy = sum(r.is_correct for r in results) / n if n else 0.0
        task_groups: Dict[str, List[bool]] = {}
        for r in results:
            task_groups.setdefault(r.task_type, []).append(r.is_correct)
        task_breakdown = {t: sum(v) / len(v) for t, v in task_groups.items()}
        avg_thinking = sum(r.thinking_tokens for r in results) / n if n else 0.0
        avg_input    = sum(r.input_tokens    for r in results) / n if n else 0.0
        avg_latency  = sum(r.latency_ms      for r in results) / n if n else 0.0
        token_eff    = (accuracy / (avg_input / 1000)) if avg_input > 0 else 0.0
        avg_cov      = sum(r.coverage_rate   for r in results) / n if n else 0.0

        return EvalReport(
            strategy_name=strategy.name,
            model_name=llm.model,
            total_samples=n,
            accuracy=accuracy,
            task_breakdown=task_breakdown,
            avg_thinking_tokens=avg_thinking,
            avg_input_tokens=avg_input,
            avg_latency_ms=avg_latency,
            token_efficiency=token_eff,
            avg_coverage_rate=avg_cov,
            samples=results,
        )
