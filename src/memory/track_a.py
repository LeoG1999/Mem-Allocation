"""Track A generates session summaries, selects them by query relevance, and
emits them in chronological order.

It supports the Split, Track-A endpoint, masking, allocation, and coverage
experiments in the paper.
"""

import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.models.llm_client import LLMClient


SESSION_SUMMARY_PROMPT = """You will receive a conversation session. Generate a memory summary for long-term retrieval.

Requirements:
1. Preserve specific facts, dates, time cues, preferences, commitments, plans, and final states.
2. If old information was overridden by new information, write only the latest version explicitly.
3. Be specific and concrete. Avoid vague generalities.
4. Output 4-6 bullet points, each specific enough to directly support question answering.

Session content:
{session_text}
"""


def pack_session_summaries(summaries, ranked_ids, budget_chars, max_sessions=None):
    """Select whole summaries in retrieval order, then serialize in conversation order."""
    header = "[Episodic Memory]\n"
    footer = "\n[/Episodic Memory]"
    blocks = {sid: f"[Session {index}]\n{text}" for index, (sid, text)
              in enumerate(summaries.items(), 1) if text}
    selected = set()
    used = len(header) + len(footer)
    for sid in ranked_ids:
        if max_sessions is not None and len(selected) >= max_sessions:
            break
        cost = len(blocks[sid]) + int(bool(selected))
        if budget_chars is None or used + cost <= budget_chars:
            selected.add(sid)
            used += cost
    ordered = [sid for sid in blocks if sid in selected]
    context = header + "\n".join(blocks[sid] for sid in ordered) + footer if ordered else ""
    return context, {
        "cap_chars": budget_chars,
        "selected_unit_count": len(ordered),
        "selected_payload_chars": sum(len(summaries[sid]) for sid in ordered),
        "serialized_chars": len(context),
        "selected_session_ids": ordered,
    }


def format_summary_session(turns, max_chars_per_turn=None):
    lines = []
    for index, (_, user, assistant) in enumerate(turns):
        lines.append(f"[Turn {index}]")
        if user:
            lines.append(f"User: {user[:max_chars_per_turn]}")
        if assistant:
            lines.append(f"Assistant: {assistant[:max_chars_per_turn]}")
    return "\n".join(lines)


def summary_cache_key(client, session_text: str, output_tokens: int) -> str:
    model = getattr(client, "api_model", None) or getattr(client, "model", "unknown")
    key = json.dumps([model, SESSION_SUMMARY_PROMPT, 0.1, output_tokens, session_text], ensure_ascii=False)
    return hashlib.sha256(key.encode()).hexdigest()[:24]


@dataclass
class SessionState:
    session_id: str
    # Raw turns stored during write phase: [(turn_idx, user, assistant), ...]
    turns: List[Tuple[int, str, str]] = field(default_factory=list)
    # Summary of the completed session.
    summary: str = ""
    summary_ready: bool = False  # True once LLM summary has been generated


class EpisodicTrack:
    """Track-A session summaries and query-aware packing."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_llm_summary: bool = True,
        max_chars_per_session: Optional[int] = None,
        memo_in_context: int = 5,
        summary_cache_dir: Optional[str] = None,
    ):
        self.llm = llm_client
        self.use_llm_summary = use_llm_summary
        self.max_chars_per_session = max_chars_per_session
        self.memo_in_context = memo_in_context
        self._summary_cache_dir = summary_cache_dir
        if summary_cache_dir:
            os.makedirs(summary_cache_dir, exist_ok=True)

        self._states: Dict[str, SessionState] = {}
        self._session_order: List[str] = []
        # Table 6 reads these packing statistics directly.
        self.last_packing_stats: Dict[str, object] = {}

    # Internal helpers

    def _get_state(self, session_id: str) -> SessionState:
        if session_id not in self._states:
            self._states[session_id] = SessionState(session_id=session_id)
            self._session_order.append(session_id)
        return self._states[session_id]

    def _build_session_text(self, state: SessionState) -> str:
        """Join all session turns and truncate to max_chars_per_session."""
        return format_summary_session(state.turns)[:self.max_chars_per_session]

    def _cache_key(self, session_text: str) -> str:
        return summary_cache_key(self.llm, session_text, 300)

    def _load_cached_summary(self, cache_key: str) -> Optional[str]:
        if not self._summary_cache_dir:
            return None
        path = os.path.join(self._summary_cache_dir, f"{cache_key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("summary")
        return None

    def _save_cached_summary(self, cache_key: str, summary: str) -> None:
        if not self._summary_cache_dir:
            return
        path = os.path.join(self._summary_cache_dir, f"{cache_key}.json")
        pending = f"{path}.{uuid.uuid4().hex}.pending"
        with open(pending, "w") as f:
            json.dump({"summary": summary}, f)
        os.replace(pending, path)

    def _summarize_session(self, state: SessionState) -> str:
        session_text = self._build_session_text(state)
        if not session_text:
            return ""

        cache_key = self._cache_key(session_text)
        cached = self._load_cached_summary(cache_key)
        if cached is not None:
            return cached

        if self.use_llm_summary:
            if self.llm is None:
                raise ValueError("Track A summary generation requires an LLM client")
            prompt = SESSION_SUMMARY_PROMPT.format(session_text=session_text)
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            summary = resp.content.strip()
            self._save_cached_summary(cache_key, summary)
            return summary

        recent = state.turns[-self.memo_in_context:]
        snippets = [
            f"[Turn {t}] U: {u[:80]} | A: {a[:80]}"
            for t, u, a in recent
        ]
        return "\n".join(snippets)

    @property
    def has_summary_cache(self) -> bool:
        return self.use_llm_summary and bool(self._summary_cache_dir)

    def prepare_summaries(self) -> None:
        """Build or load summaries of completed sessions before query evaluation."""
        pending = [
            (sid, self._states[sid])
            for sid in self._session_order
            if sid in self._states and not self._states[sid].summary_ready
        ]
        if not pending:
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        groups = {}
        for sid, state in pending:
            key = self._build_session_text(state) if self.has_summary_cache else sid
            groups.setdefault(key, []).append(state)
        with ThreadPoolExecutor(max_workers=min(8, len(groups))) as executor:
            futures = {executor.submit(self._summarize_session, states[0]): states for states in groups.values()}
            for future in as_completed(futures):
                summary = future.result()
                for state in futures[future]:
                    state.summary = summary
                    state.summary_ready = True

    # Public interface

    def add_turn(
        self,
        session_id: str,
        turn: int,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Store one raw conversation turn without calling the LLM."""
        state = self._get_state(session_id)
        state.turns.append((turn, user_content, assistant_content))
        # A newly written turn invalidates the existing summary.
        state.summary_ready = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9'_]+", text.lower()) if len(t) >= 2]

    def _bm25_rank_sessions(self, session_order: List[str], query: str) -> List[str]:
        """Rank all known sessions by BM25 relevance, most relevant first."""
        if not query:
            return session_order
        query_terms = self._tokenize(query)
        if not query_terms:
            return session_order

        docs: Dict[str, List[str]] = {}
        for sid in session_order:
            state = self._states.get(sid)
            if not state:
                docs[sid] = []
                continue
            text = " ".join(
                f"{u} {a}" for _, u, a in state.turns
            )
            docs[sid] = self._tokenize(text)

        n = len(session_order)
        df: Counter = Counter()
        for tokens in docs.values():
            df.update(set(tokens))
        idf = {
            term: math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            for term in set(query_terms)
        }
        avg_len = sum(len(t) for t in docs.values()) / max(n, 1)
        k1, b = 1.5, 0.75

        scores: Dict[str, float] = {}
        for sid, tokens in docs.items():
            tf = Counter(tokens)
            doc_len = len(tokens)
            norm = k1 * (1 - b + b * doc_len / max(avg_len, 1e-6))
            scores[sid] = sum(
                idf.get(term, 0.0) * (tf.get(term, 0) * (k1 + 1))
                / (tf.get(term, 0) + norm)
                for term in query_terms
                if tf.get(term, 0) > 0
            )

        return sorted(session_order, key=lambda s: scores.get(s, 0.0), reverse=True)

    def get_context(
        self,
        session_id: str,
        query: str = "",
        budget_chars: Optional[int] = None,
    ) -> str:
        """
        Aggregate Track-A context across all known sessions.

        Evaluation prepares summaries before query-time retrieval.

        Query-aware session selection:
          1. Rank sessions with BM25 and select the most relevant sessions that fit.
          2. Present selected sessions chronologically to preserve the timeline.

        Format:
          [Episodic Memory]
          [Session 1] <bullet summary>
          [Session 2] <bullet summary>
          ...
          [/Episodic Memory]
        """
        summaries = {}
        if budget_chars != 0:
            self.prepare_summaries()
            summaries = {sid: self._states[sid].summary for sid in self._session_order}
        candidates = [sid for sid, text in summaries.items() if text]
        ranked = self._bm25_rank_sessions(candidates, query) if query else reversed(candidates)
        context, self.last_packing_stats = pack_session_summaries(summaries, ranked, budget_chars)
        return context

    def get_summary(self, session_id: str) -> str:
        state = self._states.get(session_id)
        return state.summary if state else ""

    def clear(self, session_id: str) -> None:
        self._states.pop(session_id, None)
        if session_id in self._session_order:
            self._session_order.remove(session_id)
