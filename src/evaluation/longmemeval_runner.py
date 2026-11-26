"""
LongMemEval evaluation runner.

Data format (ICLR 2025, arXiv:2410.10813): each sample contains a multi-turn
history, one question, and a reference answer. The five task families are:
  single_session_user  -- information extraction from one session
  multi_session        -- reasoning across sessions
  temporal             -- temporal and date reasoning
  knowledge_update     -- replacement of stale information by newer information
  abstention           -- the answer is absent from history and should be refused

Usage:
    runner = LongMemEvalRunner(data_path="data/longmemeval/longmemeval_s.json")
    results = runner.evaluate(strategy, max_samples=50)
"""

import gc
import json
import os
import re
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.models.llm_client import LLMClient, LLMResponse
from src.memory.baselines.strategies import BaseMemoryStrategy, FullContextStrategy


ANSWER_PROMPT = """You have access to the following memory context. Answer the question based on this information.
If the memory context does not contain sufficient information to answer the question, respond with "I don't know" or "Not enough information."

{memory_context}

Question: {question}

Provide a concise answer directly. Do not explain your reasoning."""

JUDGE_PROMPT = """Determine whether the model's answer matches the reference answer (semantic equivalence is sufficient; exact match is not required).

Reference answer: {gold}
Model answer: {prediction}

Reply with only "YES" or "NO"."""

JUDGE_PROMPT_ABSTENTION = """I will give you an unanswerable question, an explanation, and a response from a model. Reply "YES" if the response correctly identifies the question as unanswerable. The response may state that the information is incomplete or that the available information does not contain what is asked.

Question: {question}
Explanation: {gold}
Model response: {prediction}

Does the model correctly identify the question as unanswerable? Reply with only "YES" or "NO"."""


def parse_binary_verdict(text: str) -> bool:
    """Parse the first YES/NO token returned by the judge."""
    match = re.search(r"\b(YES|NO)\b", (text or "").upper())
    if not match:
        raise ValueError(f"Judge verdict is not YES/NO: {text!r}")
    return match.group(1) == "YES"


MERGED_SESSION_ID = "__longmemeval_merged__"


@dataclass
class EvalSample:
    sample_id: str
    question: str
    gold_answer: str
    task_type: str
    history: List[Dict]     # Flat message list; session structures are flattened on load
    session_ids: List[str]  # Session IDs represented in the history


@dataclass
class SampleResult:
    sample_id: str
    task_type: str
    prediction: str
    gold_answer: str
    is_correct: bool
    memory_context_len: int = 0
    input_tokens: int = 0
    thinking_tokens: int = 0
    latency_ms: float = 0.0
    budget_chars: Optional[int] = None
    coverage_rate: float = 0.0   # injected sessions / total sessions for this sample
    # Orthogonality-analysis fields populated by DualTrackMemory only.
    attribution: Optional[dict] = None
    # Full context is stored only for length-matching and coverage experiments.
    memory_context: Optional[str] = None
    packing_stats: Optional[dict] = None
    judge_verdict_raw: str = ""


@dataclass
class EvalReport:
    strategy_name: str
    model_name: str
    total_samples: int
    accuracy: float
    task_breakdown: Dict[str, float] = field(default_factory=dict)
    avg_thinking_tokens: float = 0.0
    avg_input_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    token_efficiency: float = 0.0   # accuracy per 1K input tokens
    avg_coverage_rate: float = 0.0  # fraction of sessions injected into context (0-1)
    samples: List[SampleResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Strategy: {self.strategy_name} | Model: {self.model_name}",
            f"Accuracy: {self.accuracy:.3f} ({int(self.accuracy * self.total_samples)}/{self.total_samples})",
            f"Avg thinking_tokens: {self.avg_thinking_tokens:.0f}",
            f"Avg input_tokens: {self.avg_input_tokens:.0f}",
            f"Token efficiency: {self.token_efficiency:.4f} (acc/1K tokens)",
            "Task breakdown:",
        ]
        for task, acc in sorted(self.task_breakdown.items()):
            lines.append(f"  {task}: {acc:.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "accuracy": self.accuracy,
            "task_breakdown": self.task_breakdown,
            "avg_thinking_tokens": self.avg_thinking_tokens,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "token_efficiency": self.token_efficiency,
            "avg_coverage_rate": self.avg_coverage_rate,
            "samples": [asdict(s) for s in self.samples],
        }


def session_occurrence_ids(source_ids):
    if len(set(source_ids)) == len(source_ids):
        return list(source_ids)
    return [f"{index}:{source}" for index, source in enumerate(source_ids)]


def iter_history_turns(history: List[dict]):
    """Keep adjacent exchanges together and emit unpaired messages on their own side."""
    pending = None
    for message in history:
        if not isinstance(message, dict):
            continue
        sid = message.get("session_id", "default")
        role = message.get("role")
        content = message.get("content", "")
        if pending is not None:
            if pending["session_id"] == sid and role in ("assistant", "ai"):
                pending["assistant"] = content
                yield pending
                pending = None
                continue
            yield pending
            pending = None
        if role in ("user", "human"):
            pending = {"session_id": sid, "user": content, "assistant": ""}
        elif role in ("assistant", "ai"):
            yield {"session_id": sid, "user": "", "assistant": content}
    if pending is not None:
        yield pending


def summary_view(strategy):
    memory = getattr(strategy, "_mem", None)
    view = memory.track_a if hasattr(memory, "track_a") else strategy
    return view if callable(getattr(view, "prepare_summaries", None)) else None


def prepare_summary_cache(view, samples):
    """Prepare every selected history before the first answer or retrieval call."""
    if view is None or not view.has_summary_cache:
        return
    seen_histories = set()
    for sample in samples:
        history_id = id(sample.history)
        if history_id in seen_histories:
            continue
        seen_histories.add(history_id)
        session_ids = set()
        try:
            for turn, pair in enumerate(iter_history_turns(sample.history)):
                sid = pair["session_id"]
                if sid not in session_ids:
                    view.clear(sid)
                    session_ids.add(sid)
                view.add_turn(sid, turn, pair["user"], pair["assistant"])
            view.prepare_summaries()
        finally:
            for sid in session_ids:
                view.clear(sid)


class LongMemEvalRunner:
    def __init__(
        self,
        data_path: str,
        judge_llm: Optional[LLMClient] = None,
        answer_llm: Optional[LLMClient] = None,
        preload_max: Optional[int] = None,
    ):
        self.data_path = data_path
        self.judge_llm = judge_llm  # Judges answer correctness
        self.answer_llm = answer_llm  # Generates final answers
        self.preload_max = preload_max  # Load only the first N samples; None loads all

        self._data: Optional[List[EvalSample]] = None

    def _load_data(self) -> List[EvalSample]:
        """Load LongMemEval JSON data, optionally limited by preload_max."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"LongMemEval data not found: {self.data_path}\n"
                "Prepare the LongMemEval-S dataset as described in DATA.md"
            )

        with open(self.data_path) as f:
            raw = json.load(f)

        # Keep only requested samples and release the rest early.
        if self.preload_max is not None:
            raw = raw[: self.preload_max]

        samples = []
        for item in raw:
            history, session_ids = self._normalize_history(item)
            sample = EvalSample(
                sample_id=str(item.get("id", item.get("question_id", ""))),
                question=item.get("question", ""),
                gold_answer=str(item.get("answer", "")),
                task_type=self._normalize_task_type(item),
                history=history,
                session_ids=session_ids,
            )
            samples.append(sample)

        del raw
        return samples

    @staticmethod
    def _parse_haystack_date(raw: Any) -> datetime:
        if not raw:
            return datetime.min
        text = str(raw)
        for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return datetime.min

    @staticmethod
    def _normalize_history(item: dict) -> tuple[List[Dict], List[str]]:
        """
        Support two input layouts:
        1. history: a flat message list
        2. sessions: a multi-session structure
        """
        if isinstance(item.get("history"), list):
            history = item.get("history", [])
            session_ids = item.get("session_ids", ["default"])
            return history, session_ids

        if isinstance(item.get("haystack_sessions"), list):
            flat_history: List[Dict] = []
            session_ids: List[str] = []
            sessions = item.get("haystack_sessions", [])
            raw_session_ids = item.get("haystack_session_ids", [])
            raw_dates = item.get("haystack_dates", [])
            session_rows = []
            sources = [str(raw_session_ids[i]) if i < len(raw_session_ids) else f"session_{i}"
                       for i in range(len(sessions))]
            for idx, (sid, session) in enumerate(zip(session_occurrence_ids(sources), sessions)):
                dt = LongMemEvalRunner._parse_haystack_date(raw_dates[idx] if idx < len(raw_dates) else None)
                session_rows.append((dt, idx, sid, session))

            # Sort explicitly by time so downstream baselines preserve recency.
            for _, _, sid, session in sorted(session_rows, key=lambda row: (row[0], row[1])):
                session_ids.append(sid)
                if not isinstance(session, list):
                    continue
                for msg in session:
                    if isinstance(msg, dict):
                        msg = dict(msg)
                        msg["session_id"] = sid
                        flat_history.append(msg)
            return flat_history, (session_ids or ["default"])

        raw_sessions = item.get("sessions", [])
        if not isinstance(raw_sessions, list):
            return [], ["default"]

        flat_history: List[Dict] = []
        session_ids: List[str] = []
        sessions = [(str(s.get("session_id", s.get("id", f"session_{i}"))), s)
                    for i, s in enumerate(raw_sessions) if isinstance(s, dict)]
        occurrence_ids = session_occurrence_ids([sid for sid, _ in sessions])
        for sid, (_, session) in zip(occurrence_ids, sessions):
            session_ids.append(sid)
            messages = session.get("history", session.get("messages", []))
            if not isinstance(messages, list):
                continue
            for msg in messages:
                if isinstance(msg, dict):
                    msg = dict(msg)
                    msg["session_id"] = sid
                    flat_history.append(msg)

        return flat_history, (session_ids or ["default"])

    @staticmethod
    def _normalize_task_type(item: dict) -> str:
        raw = item.get("type", item.get("category", item.get("question_type", "unknown")))
        if not isinstance(raw, str):
            raw = str(raw)
        mapping = {
            "single-session-user": "single_session_user",
            "single-session-assistant": "single_session_assistant",
            "single-session-preference": "single_session_preference",
            "temporal-reasoning": "temporal",
            "knowledge-update": "knowledge_update",
            "multi-session": "multi_session",
        }
        if str(item.get("question_id", "")).endswith("_abs"):
            return "abstention"
        return mapping.get(raw, raw.replace("-", "_"))

    def judge_answer_with_verdict(
        self,
        prediction: str,
        gold: str,
        task_type: str = "",
        question: str = "",
    ) -> tuple[bool, str]:
        """Return correctness and the raw verdict under the paper's judge protocol."""
        if not prediction:
            return False, "EMPTY_PREDICTION"

        pred_norm = prediction.lower().strip()
        if not gold:
            return False, "EMPTY_REFERENCE"
        gold_norm = gold.lower().strip()
        if self.judge_llm is None:
            if task_type == "abstention":
                signals = (
                    "i don't know", "i do not know", "not enough information",
                    "insufficient information", "not mentioned", "not available",
                )
                matched = any(signal in pred_norm for signal in signals)
            else:
                matched = pred_norm == gold_norm or gold_norm in pred_norm
            return matched, "RULE_MATCH" if matched else "RULE_NO_MATCH"

        if task_type == "abstention":
            prompt = JUDGE_PROMPT_ABSTENTION.format(
                question=question,
                gold=gold,
                prediction=prediction,
            )
        else:
            prompt = JUDGE_PROMPT.format(gold=gold, prediction=prediction)
        resp = self.judge_llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        verdict = resp.content.strip()
        return parse_binary_verdict(verdict), verdict

    def _judge_answer(
        self,
        prediction: str,
        gold: str,
        task_type: str = "",
        question: str = "",
    ) -> bool:
        return self.judge_answer_with_verdict(prediction, gold, task_type, question)[0]

    def evaluate(
        self,
        strategy: BaseMemoryStrategy,
        answer_llm: Optional[LLMClient] = None,
        max_samples: Optional[int] = None,
        task_types: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        judge_mode: str = "hybrid",
        budget_chars: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        verbose: bool = False,
        store_context: bool = False,
    ) -> EvalReport:
        """
        Run an evaluation.

        strategy: memory strategy to evaluate
        answer_llm: answer-generating LLM, overriding the constructor setting
        max_samples: maximum number of samples; None uses all samples
        task_types: optional task-type filter
        """
        if self._data is None:
            self._data = self._load_data()

        llm = answer_llm or self.answer_llm
        if llm is None:
            raise ValueError("An answer_llm is required to generate answers")

        data = self._data
        if task_types:
            data = [s for s in data if s.task_type in task_types]
        if sample_ids:
            sample_id_set = {str(x) for x in sample_ids}
            data = [s for s in data if s.sample_id in sample_id_set]
        if shuffle:
            rng = random.Random(seed)
            data = list(data)
            rng.shuffle(data)
        if max_samples:
            data = data[:max_samples]

        view = summary_view(strategy)
        prepare_summary_cache(view, data)
        results: List[SampleResult] = []

        for i, sample in enumerate(data):
            if verbose:
                print(f"[{i+1}/{len(data)}] {sample.sample_id} ({sample.task_type})")
            touched_session_ids = set(sample.session_ids)
            use_merged_session = len(sample.session_ids) > 1
            is_dual_track = hasattr(strategy, "_mem") and hasattr(strategy._mem, "get_context_with_attribution")
            if use_merged_session:
                touched_session_ids.add(MERGED_SESSION_ID)

            try:
                for sid in touched_session_ids:
                    strategy.clear(sid)

                # Replay history chronologically and preserve session boundaries when available.
                turn = 0
                current_session = sample.session_ids[0] if sample.session_ids else "default"

                full_history = isinstance(strategy, FullContextStrategy)
                turn_pairs = () if full_history else self._iter_turn_pairs(sample.history)
                for msg_pair in turn_pairs:
                    sid = msg_pair.get("session_id", current_session)
                    write_sid = MERGED_SESSION_ID if (use_merged_session and is_dual_track) else sid
                    touched_session_ids.add(write_sid)
                    # Pass the real session ID so Track A preserves cross-session chronology.
                    if is_dual_track and use_merged_session:
                        strategy.add_turn(
                            write_sid,
                            turn,
                            msg_pair["user"],
                            msg_pair["assistant"],
                            real_session_id=sid,
                        )
                    else:
                        strategy.add_turn(
                            write_sid,
                            turn,
                            msg_pair["user"],
                            msg_pair["assistant"],
                        )
                    if use_merged_session and not is_dual_track:
                        strategy.add_turn(
                            MERGED_SESSION_ID,
                            turn,
                            msg_pair["user"],
                            msg_pair["assistant"],
                        )
                    turn += 1

                # Retrieve memory context with optional budget and attribution.
                if view is not None:
                    view.prepare_summaries()
                attribution = None
                context_session_id = MERGED_SESSION_ID if use_merged_session else current_session
                if full_history:
                    mem_ctx = strategy.render_history(sample.history)
                elif is_dual_track:
                    mem_ctx, attribution = strategy._mem.get_context_with_attribution(
                        context_session_id, query=sample.question, budget_chars=budget_chars
                    )
                else:
                    mem_ctx = strategy.get_context(
                        context_session_id, query=sample.question, budget_chars=budget_chars
                    )
                packing_stats = (
                    attribution if is_dual_track
                    else dict(getattr(strategy, "last_packing_stats", {}))
                )

                # Estimate coverage as referenced session IDs divided by total sessions.
                total_sess = max(len(sample.session_ids), 1)
                injected = len(set(re.findall(r"session[= ]([^\]\s,)]+)", mem_ctx or "")))
                coverage_rate = min(injected / total_sess, 1.0)

                prompt = ANSWER_PROMPT.format(
                    memory_context=mem_ctx if mem_ctx else "(no memory context)",
                    question=sample.question,
                )

                # Let API failures stop the cell so missing results are not counted as errors.
                resp: LLMResponse = llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )

                is_correct, judge_verdict_raw = self._judge_answer_with_mode(
                    prediction=resp.content,
                    gold=sample.gold_answer,
                    judge_mode=judge_mode,
                    task_type=sample.task_type,
                    question=sample.question,
                )

                results.append(SampleResult(
                    sample_id=sample.sample_id,
                    task_type=sample.task_type,
                    prediction=resp.content,
                    gold_answer=sample.gold_answer,
                    is_correct=is_correct,
                    memory_context_len=len(mem_ctx),
                    input_tokens=resp.input_tokens,
                    thinking_tokens=resp.thinking_tokens,
                    latency_ms=resp.latency_ms,
                    budget_chars=budget_chars,
                    coverage_rate=coverage_rate,
                    attribution=attribution,
                    memory_context=mem_ctx if store_context else None,
                    packing_stats=packing_stats,
                    judge_verdict_raw=judge_verdict_raw,
                ))
            finally:
                for sid in touched_session_ids:
                    strategy.clear(sid)

            # Collect cyclic garbage every ten samples, mainly for SDK model objects.
            if (i + 1) % 10 == 0:
                gc.collect()

        # Compute aggregate metrics.
        accuracy = sum(r.is_correct for r in results) / len(results) if results else 0.0

        task_breakdown: Dict[str, float] = {}
        task_groups: Dict[str, List[bool]] = {}
        for r in results:
            task_groups.setdefault(r.task_type, []).append(r.is_correct)
        for task, bools in task_groups.items():
            task_breakdown[task] = sum(bools) / len(bools)

        avg_thinking = sum(r.thinking_tokens for r in results) / len(results) if results else 0
        avg_input = sum(r.input_tokens for r in results) / len(results) if results else 0
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
        token_eff = (accuracy / (avg_input / 1000)) if avg_input > 0 else 0.0
        avg_cov = sum(r.coverage_rate for r in results) / len(results) if results else 0.0

        return EvalReport(
            strategy_name=strategy.name,
            model_name=llm.model,
            total_samples=len(results),
            accuracy=accuracy,
            task_breakdown=task_breakdown,
            avg_thinking_tokens=avg_thinking,
            avg_input_tokens=avg_input,
            avg_latency_ms=avg_latency,
            token_efficiency=token_eff,
            avg_coverage_rate=avg_cov,
            samples=results,
        )

    def _judge_answer_with_mode(
        self,
        prediction: str,
        gold: str,
        judge_mode: str,
        task_type: str = "",
        question: str = "",
    ) -> tuple[bool, str]:
        pred_norm = prediction.lower().strip()
        gold_norm = gold.lower().strip()
        if judge_mode == "exact":
            value = pred_norm == gold_norm
            return value, "EXACT_MATCH" if value else "EXACT_NO_MATCH"
        if judge_mode == "string":
            value = pred_norm == gold_norm or gold_norm in pred_norm
            return value, "STRING_MATCH" if value else "STRING_NO_MATCH"
        return self.judge_answer_with_verdict(prediction, gold, task_type, question)

    @staticmethod
    def _iter_turn_pairs(history: List[dict]):
        return list(iter_history_turns(history))
