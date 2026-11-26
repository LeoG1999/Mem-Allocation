"""
Session-oriented memory strategies used in the paper experiments.

The budget-sweep protocol uses the following interface:
    add_turn(session_id, turn, user_content, assistant_content)
    get_context(session_id, query, budget_chars) -> str
    clear(session_id)

For LongMemEval multi-session questions, the runner feeds both the original
session ids and a synthetic merged id. The session-aware baselines below ignore
the synthetic id on write and aggregate across the materialized sessions on
read.
"""

from abc import ABC, abstractmethod
import json
import math
import os
import re
import uuid
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np

from src.models.llm_client import LLMClient
from src.memory.track_b_rag import SemanticRAGTrack, TRACK_B_LABEL
from src.memory.track_a import SESSION_SUMMARY_PROMPT, format_summary_session, pack_session_summaries, summary_cache_key

# Standard budget grid for the sweep, in characters (approximately tokens * 4).
BUDGET_LADDER = [500, 1000, 2000, 3000, 4000, 6000]
MERGED_SESSION_ID = "__longmemeval_merged__"

class BaseMemoryStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def add_turn(
        self,
        session_id: str,
        turn: int,
        user_content: str,
        assistant_content: str,
    ) -> None: ...

    @abstractmethod
    def get_context(
        self,
        session_id: str,
        query: str = "",
        budget_chars: Optional[int] = None,
    ) -> str: ...

    def clear(self, session_id: str) -> None:
        pass


class SessionAwareStrategy(BaseMemoryStrategy):
    """Shared helpers for baselines that operate on session-level structure."""

    def __init__(self):
        self._session_turns: Dict[str, List[Tuple[int, str, str]]] = {}
        self._session_order: List[str] = []

    @staticmethod
    def _is_merged_session(session_id: str) -> bool:
        return session_id == MERGED_SESSION_ID

    def _material_session_ids(self, session_id: str) -> List[str]:
        if self._is_merged_session(session_id):
            return list(self._session_order)
        return [session_id] if session_id in self._session_turns else []

    def _session_number(self, session_id: str) -> int:
        """Chronological label for rendering a session without exposing its source ID."""
        return self._session_order.index(session_id) + 1

    def add_turn(self, session_id, turn, user_content, assistant_content):
        if self._is_merged_session(session_id):
            return
        if session_id not in self._session_turns:
            self._session_turns[session_id] = []
            self._session_order.append(session_id)
        self._session_turns[session_id].append((turn, user_content, assistant_content))

    def clear(self, session_id):
        if self._is_merged_session(session_id):
            return
        self._session_turns.pop(session_id, None)
        self._session_order = [sid for sid in self._session_order if sid != session_id]


# ── 0. Full Context Upper Bound ──────────────────────────────────

class FullContextStrategy(BaseMemoryStrategy):
    """
    Upper-bound reference that retains the complete raw conversation.

    This is not a deployable memory-compression strategy. It estimates an
    approximate upper bound for the current benchmark, prompt, and backbone.
    """

    name = "full_context"

    def __init__(self):
        self._turns: Dict[str, List[Tuple[int, str, str]]] = {}

    def add_turn(self, session_id, turn, user_content, assistant_content):
        self._turns.setdefault(session_id, []).append((turn, user_content, assistant_content))

    def get_context(self, session_id, query="", budget_chars=None):
        turns = self._turns.get(session_id, [])
        if not turns:
            return ""

        lines = []
        for turn, user, asst in turns:
            lines.append(f"[turn {turn}] User: {user}")
            lines.append(f"[turn {turn}] Assistant: {asst}")
        return "[Full Conversation]\n" + "\n".join(lines) + "\n[/Full Conversation]"

    @staticmethod
    def render_history(history):
        if not history:
            return ""
        lines = ["[Full Conversation]"]
        previous = object()
        session_number = 0
        for message in history:
            session = message.get("session_id")
            if session != previous:
                session_number += 1
                lines.append(f"[Session {session_number}]")
                previous = session
            lines.append(f"{message['role'].capitalize()}: {message['content']}")
        lines.append("[/Full Conversation]")
        return "\n".join(lines)

    def clear(self, session_id):
        self._turns.pop(session_id, None)


# ── 1. Recent Session Truncation ─────────────────────────────────

class TruncationStrategy(BaseMemoryStrategy):
    """Discard all history. The budget is unused and the context is always empty."""

    name = "truncation"

    def add_turn(self, session_id, turn, user_content, assistant_content):
        pass

    def get_context(self, session_id, query="", budget_chars=None):
        return ""


class RecentSessionsStrategy(SessionAwareStrategy):
    """Main baseline: retain raw text from the N most recent sessions."""

    name = "recent_sessions"

    def __init__(
        self,
        window_size: Optional[int] = None,
        max_chars_per_turn: Optional[int] = 400,
        max_turns_per_session: Optional[int] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.max_chars_per_turn = max_chars_per_turn
        self.max_turns_per_session = max_turns_per_session

    def _format_session(self, session_id: str) -> str:
        turns = self._session_turns.get(session_id, [])
        if self.max_turns_per_session is not None:
            turns = turns[-self.max_turns_per_session :]
        lines = [f"[session {self._session_number(session_id)}]"]
        for turn, user, asst in turns:
            if self.max_chars_per_turn is not None:
                user = user[: self.max_chars_per_turn]
                asst = asst[: self.max_chars_per_turn]
            lines.append(f"[turn {turn}] User: {user}")
            lines.append(f"[turn {turn}] Assistant: {asst}")
        return "\n".join(lines)

    def get_context(self, session_id, query="", budget_chars=None):
        session_ids = self._material_session_ids(session_id)
        if not session_ids:
            return ""

        recent_ids = session_ids[-self.window_size :] if self.window_size is not None else session_ids
        candidates = list(reversed(recent_ids))
        header = "[Recent Sessions]\n"
        footer = "\n[/Recent Sessions]"
        payload_limit = (
            max(0, budget_chars - len(header) - len(footer))
            if budget_chars is not None else None
        )
        blocks: List[str] = []
        total = 0
        for sid in candidates:
            block = self._format_session(sid)
            separator = 2 if blocks else 0
            if payload_limit is not None and total + separator + len(block) > payload_limit:
                continue
            blocks.insert(0, block)
            total += separator + len(block)

        if not blocks:
            return ""
        return header + "\n\n".join(blocks) + footer


# ── 2. Session Summary Only ──────────────────────────────────────

class SessionSummaryStrategy(SessionAwareStrategy):
    """Rank raw sessions and pack their cached summaries in chronological order."""

    name = "session_summary"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        recent_session_count: int = 8,
        max_chars_per_turn: Optional[int] = None,
        max_turns_per_session: Optional[int] = None,
        use_llm_summary: bool = True,
        llm_input_char_limit: Optional[int] = None,
        max_tokens_summary: int = 300,
        query_aware: bool = True,
        summary_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.llm = llm_client
        self.recent_session_count = recent_session_count
        self.max_chars_per_turn = max_chars_per_turn
        self.max_turns_per_session = max_turns_per_session
        self.use_llm_summary = use_llm_summary
        self.llm_input_char_limit = llm_input_char_limit
        self.max_tokens_summary = max_tokens_summary
        self.query_aware = query_aware
        self._summary_cache: Dict[str, str] = {}
        self._disk_cache_dir = summary_cache_dir
        if summary_cache_dir:
            os.makedirs(summary_cache_dir, exist_ok=True)

    def _format_session_for_summary(self, session_id: str) -> str:
        turns = self._session_turns.get(session_id, [])
        return format_summary_session(turns, self.max_chars_per_turn)[:self.llm_input_char_limit]

    def _extractive_summary(self, session_id: str) -> str:
        turns = self._session_turns.get(session_id, [])
        if not turns:
            return ""
        max_turns = self.max_turns_per_session or len(turns)
        chosen = turns[:1]
        if len(turns) > 2 and max_turns > 1:
            chosen.extend(turns[-(max_turns - 1) :])
        elif len(turns) > 1:
            chosen = turns[-max_turns:]

        lines = []
        for turn, user, asst in chosen:
            if self.max_chars_per_turn is not None:
                user = user[:self.max_chars_per_turn]
                asst = asst[:self.max_chars_per_turn]
            lines.append(f"- user[{turn}]: {user}")
            lines.append(f"- assistant[{turn}]: {asst}")
        return "\n".join(lines)

    def _disk_cache_key(self, session_text: str) -> str:
        return summary_cache_key(self.llm, session_text, self.max_tokens_summary)

    def _load_disk_cache(self, key: str) -> Optional[str]:
        if not self._disk_cache_dir:
            return None
        path = os.path.join(self._disk_cache_dir, f"{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("summary")
        return None

    def _save_disk_cache(self, key: str, summary: str) -> None:
        if not self._disk_cache_dir:
            return
        path = os.path.join(self._disk_cache_dir, f"{key}.json")
        pending = f"{path}.{uuid.uuid4().hex}.pending"
        with open(pending, "w") as f:
            json.dump({"summary": summary}, f)
        os.replace(pending, path)

    @property
    def has_summary_cache(self) -> bool:
        return self.use_llm_summary and bool(self._disk_cache_dir)

    def prepare_summaries(self) -> None:
        for sid in self._session_order:
            self._summarize_session(sid)

    def _summarize_session(self, session_id: str) -> str:
        if session_id in self._summary_cache:
            return self._summary_cache[session_id]

        if not self.use_llm_summary or self.llm is None:
            summary = self._extractive_summary(session_id)
            self._summary_cache[session_id] = summary
            return summary

        session_text = self._format_session_for_summary(session_id)
        disk_key = self._disk_cache_key(session_text)
        cached = self._load_disk_cache(disk_key)
        if cached is not None:
            self._summary_cache[session_id] = cached
            return cached

        prompt = SESSION_SUMMARY_PROMPT.format(session_text=session_text)
        resp = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=self.max_tokens_summary,
        )
        summary = resp.content.strip()
        self._summary_cache[session_id] = summary
        self._save_disk_cache(disk_key, summary)
        return summary

    def _bm25_rank_sessions(self, session_ids: List[str], query: str) -> List[str]:
        """Return session_ids ranked by BM25 relevance to query (most relevant first)."""
        if not query:
            return session_ids

        def tokenize(text: str) -> List[str]:
            return [t for t in re.findall(r"[a-z0-9'_]+", text.lower()) if len(t) >= 2]

        query_terms = tokenize(query)
        if not query_terms:
            return session_ids

        # Build per-session document text and token counts
        docs: Dict[str, List[str]] = {}
        for sid in session_ids:
            turns = self._session_turns.get(sid, [])
            lines = []
            for _, user, asst in turns:
                lines.append(user[:self.max_chars_per_turn])
                lines.append(asst[:self.max_chars_per_turn])
            docs[sid] = tokenize(" ".join(lines))

        n = len(session_ids)
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
            score = sum(
                idf.get(term, 0.0) * (tf.get(term, 0) * (k1 + 1))
                / (tf.get(term, 0) + norm)
                for term in query_terms
                if tf.get(term, 0) > 0
            )
            scores[sid] = score

        return sorted(session_ids, key=lambda s: scores.get(s, 0.0), reverse=True)

    def get_context(self, session_id, query="", budget_chars=None):
        session_ids = self._material_session_ids(session_id)
        summaries = {sid: self._summarize_session(sid) for sid in session_ids} if budget_chars != 0 else {}
        candidates = [sid for sid, text in summaries.items() if text]
        ranked = (self._bm25_rank_sessions(candidates, query) if self.query_aware and query
                  else reversed(candidates))
        context, self.last_packing_stats = pack_session_summaries(
            summaries, ranked, budget_chars, self.recent_session_count
        )
        return context

    def clear(self, session_id):
        if self._is_merged_session(session_id):
            return
        self._summary_cache.pop(session_id, None)
        super().clear(session_id)


# ── 3. Session Retrieval Appendix Baseline ──────────────────────

class BM25SessionStrategy(SessionAwareStrategy):
    """Full-session BM25 retrieval used in Table 1."""

    name = "bm25_session"

    def __init__(
        self,
        top_k: Optional[int] = None,
        max_chars_per_turn: int = 220,
        recency_bias: float = 0.0,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        super().__init__()
        self.top_k = top_k
        self.max_chars_per_turn = max_chars_per_turn
        self.recency_bias = recency_bias
        self.k1 = k1
        self.b = b
        self._session_docs: Dict[str, str] = {}
        self._session_terms: Dict[str, List[str]] = {}
        self._session_tf: Dict[str, Counter[str]] = {}
        self._session_len: Dict[str, int] = {}

    def _session_document(self, session_id: str) -> str:
        turns = self._session_turns.get(session_id, [])
        lines = []
        for turn, user, asst in turns:
            lines.append(f"[turn {turn}] User: {user[:self.max_chars_per_turn]}")
            lines.append(f"[turn {turn}] Assistant: {asst[:self.max_chars_per_turn]}")
        return "\n".join(lines)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in re.findall(r"[a-z0-9'_]+", text.lower()) if len(tok) >= 2]

    def _ensure_materialized(self, session_id: str) -> None:
        if session_id in self._session_docs:
            return
        doc = self._session_document(session_id)
        self._session_docs[session_id] = doc
        terms = self._tokenize(doc)
        self._session_terms[session_id] = terms
        self._session_tf[session_id] = Counter(terms)
        self._session_len[session_id] = len(terms)

    def _bm25_score(
        self,
        session_id: str,
        query_terms: List[str],
        idf: Dict[str, float],
        avg_doc_len: float,
        recency_rank: int,
    ) -> float:
        self._ensure_materialized(session_id)
        tf = self._session_tf[session_id]
        doc_len = self._session_len[session_id]
        score = 0.0
        norm = self.k1 * (1 - self.b + self.b * doc_len / max(avg_doc_len, 1e-6))
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            score += idf.get(term, 0.0) * (freq * (self.k1 + 1)) / (freq + norm)
        return score + self.recency_bias * recency_rank

    def get_context(self, session_id, query="", budget_chars=None):
        target_sessions = self._material_session_ids(session_id)
        if not target_sessions:
            return ""
        for sid in target_sessions:
            self._ensure_materialized(sid)
        query_terms = self._tokenize(query)
        if not query_terms:
            query_terms = ["general"]
        doc_count = len(target_sessions)
        avg_doc_len = sum(self._session_len[sid] for sid in target_sessions) / max(doc_count, 1)
        df = Counter()
        for sid in target_sessions:
            df.update(set(self._session_terms[sid]))
        idf = {
            term: math.log(1 + (doc_count - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            for term in set(query_terms)
        }
        scored = []
        for idx, sid in enumerate(reversed(target_sessions), start=1):
            score = self._bm25_score(sid, query_terms, idf, avg_doc_len, recency_rank=idx)
            scored.append((score, sid))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)
        header = "[Session Retrieval]\n"
        footer = "\n[/Session Retrieval]"
        selected = []
        total = 0
        limit = (
            max(0, budget_chars - len(header) - len(footer))
            if budget_chars is not None else 4000
        )
        for score, sid in ranked:
            if self.top_k is not None and len(selected) >= self.top_k:
                break
            self._ensure_materialized(sid)
            block = f"[session {self._session_number(sid)}]\n{self._session_docs[sid]}"
            separator = 1 if selected else 0
            if total + separator + len(block) > limit:
                continue
            selected.append((sid, block))
            total += separator + len(block)

        if not selected:
            return ""
        order = {sid: index for index, sid in enumerate(target_sessions)}
        lines = [block for _, block in sorted(selected, key=lambda row: order[row[0]])]
        return header + "\n".join(lines) + footer

    def clear(self, session_id):
        if self._is_merged_session(session_id):
            return
        self._session_docs.pop(session_id, None)
        self._session_terms.pop(session_id, None)
        self._session_tf.pop(session_id, None)
        self._session_len.pop(session_id, None)
        super().clear(session_id)


# ── 4. Embedding Session Strategy ────────────────────────────────

class EmbeddingSessionStrategy(SessionAwareStrategy):
    """Session-level dense retrieval baseline using sentence-transformers.

    Encodes each session as a single embedding vector (mean-pooled over turns),
    then retrieves top-k sessions by cosine similarity to the query embedding.
    Fills the budget with raw session text, identical to BM25SessionStrategy,
    so the two strategies differ only in the scoring signal: lexical vs semantic.

    Model default: all-MiniLM-L6-v2 (local, 80 MB, 384-dim, English-focused).
    """

    name = "embedding_session"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 8,
        max_chars_per_turn: int = 220,
        recency_bias: float = 0.02,
    ):
        super().__init__()
        self.top_k = top_k
        self.max_chars_per_turn = max_chars_per_turn
        self.recency_bias = recency_bias
        self._model_name = model_name
        self._model = None  # lazy load on first get_context call
        self._session_docs: Dict[str, str] = {}
        self._session_vecs: Dict[str, "np.ndarray"] = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _session_document(self, session_id: str) -> str:
        turns = self._session_turns.get(session_id, [])
        lines = []
        for turn, user, asst in turns:
            lines.append(f"[turn {turn}] User: {user[:self.max_chars_per_turn]}")
            lines.append(f"[turn {turn}] Assistant: {asst[:self.max_chars_per_turn]}")
        return "\n".join(lines)

    def _ensure_encoded(self, session_id: str) -> None:
        if session_id in self._session_vecs:
            return
        import numpy as np
        doc = self._session_document(session_id)
        self._session_docs[session_id] = doc
        vec = self._get_model().encode(doc, normalize_embeddings=True)
        self._session_vecs[session_id] = vec

    def _cosine_score(self, session_id: str, query_vec: "np.ndarray", recency_rank: int) -> float:
        import numpy as np
        sim = float(np.dot(self._session_vecs[session_id], query_vec))
        return sim + self.recency_bias * recency_rank

    def get_context(self, session_id, query="", budget_chars=None):
        import numpy as np
        target_sessions = self._material_session_ids(session_id)
        if not target_sessions:
            return ""
        for sid in target_sessions:
            self._ensure_encoded(sid)

        query_text = query if query else "general conversation"
        query_vec = self._get_model().encode(query_text, normalize_embeddings=True)

        scored = []
        for idx, sid in enumerate(reversed(target_sessions), start=1):
            score = self._cosine_score(sid, query_vec, recency_rank=idx)
            scored.append((score, sid))

        selected = sorted(scored, key=lambda item: item[0], reverse=True)[: self.top_k]
        lines = []
        total = 0
        limit = budget_chars if budget_chars is not None else 4000
        for score, sid in selected:
            block = f"- (session={self._session_number(sid)}, score={score:.3f}) {self._session_docs[sid]}"
            remaining = limit - total
            if remaining <= 0:
                break
            if len(block) > remaining:
                block = block[:remaining]
            lines.append(block)
            total += len(block)

        if not lines:
            return ""
        return "[Session Retrieval (Embedding)]\n" + "\n".join(lines) + "\n[/Session Retrieval (Embedding)]"

    def clear(self, session_id):
        if self._is_merged_session(session_id):
            return
        self._session_docs.pop(session_id, None)
        self._session_vecs.pop(session_id, None)
        super().clear(session_id)


# ── 5. Embedding Chunk Strategy ──────────────────────────────────

class _WordChunkStrategy(SessionAwareStrategy):
    """Common word windows, budget accounting, and output order for both retrievers."""

    def _split_chunks(self, text: str, turn_idx: int) -> List[tuple]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if not words:
            return []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk_words = words[start: start + self.chunk_size]
            chunks.append((" ".join(chunk_words), turn_idx))
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def _pack_scored(self, scored, budget_chars):
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        header = "[Chunk Retrieval]\n"
        footer = "\n[/Chunk Retrieval]"
        selected = []
        total = 0
        limit = (
            max(0, budget_chars - len(header) - len(footer))
            if budget_chars is not None else 4000
        )
        for _, turn, sid, chunk_text, source_index in ranked:
            if len(selected) >= self.top_k:
                break
            block = f"[turn {turn}, session {self._session_number(sid)}] {chunk_text}"
            separator = 1 if selected else 0
            if total + separator + len(block) > limit:
                continue
            selected.append(((turn, source_index), block))
            total += separator + len(block)

        if not selected:
            return ""
        lines = [block for _, block in sorted(selected, key=lambda row: row[0])]
        return header + "\n".join(lines) + footer


class EmbeddingChunkStrategy(_WordChunkStrategy):
    """Chunk-level dense retrieval baseline — the standard SOTA embedding approach.

    Each session is split into overlapping fixed-size chunks (measured in words).
    Every chunk is encoded independently so the full session content fits within
    the encoder's token limit (256 for MiniLM).  Retrieval scores the query
    against each chunk; the top-k chunks are selected and injected in
    chronological order, mirroring Track B's BM25 chunk pipeline.

    BM25ChunkStrategy uses the same word windows and context serializer.

    Default model: all-MiniLM-L6-v2 (local, 80 MB, 384-dim).
    chunk_size=64 words ≈ 80-100 tokens — fits well within 256-token limit.
    """

    name = "embedding_chunk"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 8,
        chunk_size: int = 64,       # words per chunk
        chunk_overlap: int = 16,    # word overlap between consecutive chunks
        recency_bias: float = 0.02,
    ):
        super().__init__()
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recency_bias = recency_bias
        self._model_name = model_name
        self._model = None
        # chunk storage: session_id → list of (chunk_text, turn_idx, chunk_global_idx)
        self._chunks: Dict[str, List[tuple]] = {}
        self._chunk_vecs: Dict[str, "np.ndarray"] = {}  # key = f"{sid}:{ci}"

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def add_turn(self, session_id, turn, user_content, assistant_content):
        if self._is_merged_session(session_id):
            return
        if session_id not in self._session_turns:
            self._session_turns[session_id] = []
            self._session_order.append(session_id)
        self._session_turns[session_id].append((turn, user_content, assistant_content))
        # Index new chunks immediately
        if session_id not in self._chunks:
            self._chunks[session_id] = []
        base = len(self._chunks[session_id])
        for text in (user_content, assistant_content):
            for chunk_text, t in self._split_chunks(text, turn):
                ci = base + len(self._chunks[session_id]) - base
                key = f"{session_id}:{len(self._chunks[session_id])}"
                self._chunks[session_id].append((chunk_text, turn, key))

    def _ensure_encoded(self, session_id: str) -> None:
        import numpy as np
        for chunk_text, turn, key in self._chunks.get(session_id, []):
            if key not in self._chunk_vecs:
                vec = self._get_model().encode(chunk_text, normalize_embeddings=True)
                self._chunk_vecs[key] = vec

    def get_context(self, session_id, query="", budget_chars=None):
        import numpy as np
        target_sessions = self._material_session_ids(session_id)
        if not target_sessions:
            return ""

        # Encode all chunks (lazy)
        for sid in target_sessions:
            self._ensure_encoded(sid)

        query_text = query if query else "general conversation"
        query_vec = self._get_model().encode(query_text, normalize_embeddings=True)

        # Score every chunk; add recency bias by global position (earlier = lower rank)
        all_chunks = []
        global_idx = 0
        for sid in target_sessions:
            for chunk_text, turn, key in self._chunks.get(sid, []):
                all_chunks.append((sid, chunk_text, turn, key, global_idx))
                global_idx += 1
        total_chunks = max(global_idx, 1)

        scored = []
        for sid, chunk_text, turn, key, gidx in all_chunks:
            vec = self._chunk_vecs.get(key)
            if vec is None:
                continue
            sim = float(np.dot(vec, query_vec))
            # Recency: chunks near the end get a small boost
            recency = self.recency_bias * (gidx / total_chunks)
            scored.append((sim + recency, turn, sid, chunk_text, gidx))

        return self._pack_scored(scored, budget_chars)

    def clear(self, session_id):
        if self._is_merged_session(session_id):
            return
        for _, _, key in self._chunks.pop(session_id, []):
            self._chunk_vecs.pop(key, None)
        super().clear(session_id)


# ── 5b. BM25 Chunk Strategy ──────────────────────────────────────

class BM25ChunkStrategy(_WordChunkStrategy):
    """Chunk-level BM25 retrieval baseline.

    Mirrors EmbeddingChunkStrategy exactly (same 64-word chunks, same top_k,
    same recency_bias) but scores with BM25 instead of cosine on MiniLM.

    Uses the same chunks and top-k as Dense 64-Word Chunks, changing only the retriever.
    """

    name = "bm25_chunk"

    def __init__(
        self,
        top_k: int = 8,
        chunk_size: int = 64,        # words per chunk
        chunk_overlap: int = 16,
        recency_bias: float = 0.02,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        super().__init__()
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recency_bias = recency_bias
        self.k1 = k1
        self.b = b
        # session_id → list of (chunk_text, turn_idx, chunk_key)
        self._chunks: Dict[str, List[tuple]] = {}
        self._chunk_tokens: Dict[str, List[str]] = {}  # key → tokens

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in re.findall(r"[a-z0-9'_]+", text.lower()) if len(tok) >= 2]

    def add_turn(self, session_id, turn, user_content, assistant_content):
        if self._is_merged_session(session_id):
            return
        if session_id not in self._session_turns:
            self._session_turns[session_id] = []
            self._session_order.append(session_id)
        self._session_turns[session_id].append((turn, user_content, assistant_content))
        if session_id not in self._chunks:
            self._chunks[session_id] = []
        for text in (user_content, assistant_content):
            for chunk_text, t in self._split_chunks(text, turn):
                key = f"{session_id}:{len(self._chunks[session_id])}"
                self._chunks[session_id].append((chunk_text, t, key))
                self._chunk_tokens[key] = self._tokenize(chunk_text)

    def get_context(self, session_id, query="", budget_chars=None):
        target_sessions = self._material_session_ids(session_id)
        if not target_sessions:
            return ""

        # Gather every chunk across the target sessions, with global ordering for recency
        all_chunks = []
        global_idx = 0
        for sid in target_sessions:
            for chunk_text, turn, key in self._chunks.get(sid, []):
                all_chunks.append((sid, chunk_text, turn, key, global_idx))
                global_idx += 1
        if not all_chunks:
            return ""
        total_chunks = max(global_idx, 1)

        query_terms = self._tokenize(query) or ["general"]

        # Document frequency over chunks (the "documents" are chunks here)
        n = len(all_chunks)
        df: Counter = Counter()
        for _, _, _, key, _ in all_chunks:
            df.update(set(self._chunk_tokens.get(key, [])))
        idf = {
            term: math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            for term in set(query_terms)
        }
        avg_len = sum(len(self._chunk_tokens.get(k, [])) for _, _, _, k, _ in all_chunks) / max(n, 1)

        scored = []
        for sid, chunk_text, turn, key, gidx in all_chunks:
            tokens = self._chunk_tokens.get(key, [])
            tf = Counter(tokens)
            doc_len = len(tokens)
            norm = self.k1 * (1 - self.b + self.b * doc_len / max(avg_len, 1e-6))
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                score += idf.get(term, 0.0) * (freq * (self.k1 + 1)) / (freq + norm)
            recency = self.recency_bias * (gidx / total_chunks)
            scored.append((score + recency, turn, sid, chunk_text, gidx))

        return self._pack_scored(scored, budget_chars)

    def clear(self, session_id):
        if self._is_merged_session(session_id):
            return
        for _, _, key in self._chunks.pop(session_id, []):
            self._chunk_tokens.pop(key, None)
        super().clear(session_id)


# ── 5b. BM25 Chunk (Track B params) — Full-Budget Allocation Control ────────

class BM25ChunkTrackBStrategy(SessionAwareStrategy):
    """Allocate the entire budget to the same chunk index and packer used by Split."""

    name = "bm25_chunk_trackb"

    track_type = SemanticRAGTrack

    def __init__(
        self, chunk_size_chars=1024, chunk_overlap_chars=128, k1=1.5, b=0.75,
        packing_mode="whole_unit_skip", budget_includes_wrapper=False,
        top_k=None, label=TRACK_B_LABEL, **index_options,
    ):
        super().__init__()
        self.budget_includes_wrapper = budget_includes_wrapper
        self.track = self.track_type(
            chunk_size=chunk_size_chars / 4, chunk_overlap=chunk_overlap_chars / 4,
            k1=k1, b=b, packing_mode=packing_mode, top_k=top_k, label=label, **index_options,
        )
        self.last_packing_stats = {}

    @property
    def packing_mode(self):
        return self.track.packing_mode

    @packing_mode.setter
    def packing_mode(self, value):
        if value not in {"whole_unit_skip", "break_on_first_oversize", "truncate_last"}:
            raise ValueError(f"Unsupported packing mode: {value}")
        self.track.packing_mode = value

    def add_turn(self, session_id, turn, user_content, assistant_content):
        if self._is_merged_session(session_id):
            return
        super().add_turn(session_id, turn, user_content, assistant_content)
        self.track.add(session_id, [{"role": "user", "content": user_content},
                                    {"role": "assistant", "content": assistant_content}], turn)

    def get_context(self, session_id, query="", budget_chars=None):
        hits = self.track.search(self._material_session_ids(session_id), query or "general context")
        cap = 4000 if budget_chars is None else budget_chars
        overhead = len(self.track.header) + len(self.track.footer) if self.budget_includes_wrapper else 0
        limit = max(0, cap - overhead)
        context = self.track.format_context_block(hits, limit)
        self.last_packing_stats = dict(self.track.last_packing_stats, cap_chars=cap, payload_cap_chars=limit)
        return context

    def clear(self, session_id):
        self.track.clear(session_id)
        super().clear(session_id)


# ── 6. DualTrack Ablation Wrappers ───────────────────────────────
#
# These thin wrappers expose the DualTrackMemory ablation modes as
# first-class strategies so experiment scripts can treat them uniformly.
# Import is deferred inside the class to avoid circular import.

class _DualTrackWrapper(BaseMemoryStrategy):
    """Base wrapper that delegates to DualTrackMemory with specific ablation flags."""

    name = "dual_track"

    def __init__(
        self,
        llm_client=None,
        persist_dir: str = "data/cache/dual_track",
        use_llm_memo: bool = True,
        use_router: bool = False,
        mask_track_a: bool = False,
        mask_track_b: bool = False,
        enable_track_a: bool = True,
        enable_track_b: bool = True,
        packing_mode: str = "whole_unit_skip",
        summary_cache_dir: Optional[str] = None,
    ):
        from src.memory.dual_track import DualTrackMemory
        self._mem = DualTrackMemory(
            llm_client=llm_client,
            persist_dir=persist_dir,
            use_llm_memo=use_llm_memo,
            use_router=use_router,
            mask_track_a=mask_track_a,
            mask_track_b=mask_track_b,
            enable_track_a=enable_track_a,
            enable_track_b=enable_track_b,
            track_b_packing_mode=packing_mode,
            summary_cache_dir=summary_cache_dir,
        )

    def add_turn(self, session_id, turn, user_content, assistant_content, real_session_id=None):
        self._mem.add_turn(session_id, turn, user_content, assistant_content, real_session_id=real_session_id)

    def get_context(self, session_id, query="", budget_chars=None):
        return self._mem.get_context(session_id, query=query, budget_chars=budget_chars)

    def clear(self, session_id):
        self._mem.clear(session_id)


class DualTrackStrategy(_DualTrackWrapper):
    """Static 50:50 dual-track strategy used in the main experiments."""
    name = "dual_track"


class DualTrackMaskAStrategy(_DualTrackWrapper):
    """A+B structure, but Track A output is masked at read time.

    Both tracks still receive the full write history (counterfactual masking).
    This isolates the causal contribution of Track A's context output.
    """
    name = "dual_track_mask_a"

    def __init__(self, llm_client=None, persist_dir="data/cache/dual_track_mask_a",
                 use_llm_memo=True, summary_cache_dir=None):
        super().__init__(
            llm_client=llm_client,
            persist_dir=persist_dir,
            use_llm_memo=use_llm_memo,
            mask_track_a=True,
            mask_track_b=False,
            summary_cache_dir=summary_cache_dir,
        )


class DualTrackMaskBStrategy(_DualTrackWrapper):
    """A+B structure, but Track B output is masked at read time.

    Both tracks still receive the full write history (counterfactual masking).
    This isolates the causal contribution of Track B's context output.
    """
    name = "dual_track_mask_b"

    def __init__(self, llm_client=None, persist_dir="data/cache/dual_track_mask_b",
                 use_llm_memo=True, summary_cache_dir=None):
        super().__init__(
            llm_client=llm_client,
            persist_dir=persist_dir,
            use_llm_memo=use_llm_memo,
            mask_track_a=False,
            mask_track_b=True,
            summary_cache_dir=summary_cache_dir,
        )
