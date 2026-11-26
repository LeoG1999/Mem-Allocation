"""Track B provides BM25 retrieval over raw-history chunks and whole-unit packing.

It supports the Split, Track-B concentration, chunk-size, and allocation
experiments in the paper.
"""

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Union

TRACK_B_LABEL = "Semantic Memory"
TRACK_B_HEADER = f"[{TRACK_B_LABEL}]\n"
TRACK_B_FOOTER = f"\n[/{TRACK_B_LABEL}]"


def split_chunk_limit(budget: int, *, token_budget: bool = False) -> Optional[int]:
    """Maximum packed chunks in a mixed allocation over the paper's main budget grid."""
    if token_budget or budget <= 6000:
        return 5
    return None


def wrapper_chars(context: str) -> int:
    for label in (TRACK_B_LABEL, "Chunk Retrieval (BM25)"):
        header, footer = f"[{label}]\n", f"\n[/{label}]"
        if context.endswith(footer) and (context.startswith(header) or "\n\n" + header in context):
            return len(header) + len(footer)
    return 0


def payload_chars(stats: dict, context: str) -> int:
    if "budgeted_chars" in stats:
        return int(stats["budgeted_chars"])
    if context:
        return len(context) - wrapper_chars(context)
    if stats.get("serialized_chars"):
        raise ValueError("Stored context or budgeted_chars is required for Track-B accounting")
    return 0


@dataclass
class MemoryEntry:
    id: str
    text: str           # Raw text span
    score: float = 0.0  # BM25 retrieval score
    turn: int = -1      # Source turn
    role: str = ""      # user / assistant
    source_index: int = 0

    @property
    def source_order(self) -> tuple[int, int]:
        return self.turn, self.source_index


class SemanticRAGTrack:
    """BM25 chunk index for Track B."""

    def __init__(
        self,
        persist_dir: str = "data/cache/track_b",
        collection_prefix: str = "trackb",
        chunk_size: int = 256,      # Approximate tokens (characters / 4)
        chunk_overlap: int = 32,
        top_k: Optional[int] = None,
        k1: float = 1.5,            # BM25 term-frequency saturation
        b: float = 0.75,            # BM25 document-length normalization
        packing_mode: str = "whole_unit_skip",
        label: str = TRACK_B_LABEL,
    ):
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        self.header, self.footer = f"[{label}]\n", f"\n[/{label}]"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        if packing_mode not in {"break_on_first_oversize", "whole_unit_skip", "truncate_last"}:
            raise ValueError(
                f"Unsupported packing mode: {packing_mode}"
            )
        self.packing_mode = packing_mode

        # Per-session chunks: session_id -> [(chunk_id, text, turn, role), ...]
        self._chunks: dict[str, list[tuple[str, str, int, str]]] = {}
        self.last_packing_stats: dict[str, object] = {}

    # Internal helpers

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple English tokenizer: lowercase alphanumerics, excluding one-character terms."""
        return [t for t in re.findall(r"[a-z0-9'_]+", text.lower()) if len(t) >= 2]

    def _chunk_text(self, text: str) -> list[str]:
        """Split by character length with overlap to reduce boundary loss."""
        char_size = int(self.chunk_size * 4)
        char_overlap = int(self.chunk_overlap * 4)
        if len(text) <= char_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + char_size
            chunk = text[start:end]
            boundary = max(chunk.rfind(". "), chunk.rfind("\n"))
            if boundary > char_size // 2:
                chunk = chunk[: boundary + 1]
            chunks.append(chunk.strip())
            if start + len(chunk) >= len(text):
                break
            advance = len(chunk) - char_overlap
            if advance <= 0:
                break  # The final span is included; the remainder is no larger than the overlap.
            start += advance
        return [c for c in chunks if c]

    def _bm25_score(
        self,
        chunk_tokens: list[str],
        query_terms: list[str],
        idf: dict[str, float],
        avg_doc_len: float,
    ) -> float:
        tf = Counter(chunk_tokens)
        doc_len = len(chunk_tokens)
        score = 0.0
        norm = self.k1 * (1 - self.b + self.b * doc_len / max(avg_doc_len, 1e-6))
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            score += idf.get(term, 0.0) * (freq * (self.k1 + 1)) / (freq + norm)
        return score

    # Public interface

    def add(
        self,
        session_id: str,
        messages: list[dict],
        turn: int = -1,
    ) -> int:
        """
        Split conversation messages into chunks and store them in the BM25 index.

        messages: [{"role": "user"|"assistant", "content": "..."}]
        Returns the number of chunks added.
        """
        if session_id not in self._chunks:
            self._chunks[session_id] = []

        added = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            tagged = f"[{role}] {content}"
            for chunk in self._chunk_text(tagged):
                self._chunks[session_id].append((str(uuid.uuid4()), chunk, turn, role))
                added += 1
        return added

    def chunks_for(self, session_id: Union[str, list[str]]) -> list:
        if isinstance(session_id, str):
            return self._chunks.get(session_id, [])
        return [chunk for sid in session_id for chunk in self._chunks.get(sid, [])]

    def search(
        self,
        session_id: Union[str, list[str]],
        query: str,
        top_k: Optional[int] = None,
    ) -> list[MemoryEntry]:
        """Return the top-k text chunks ranked by BM25 relevance to the query."""
        chunks = self.chunks_for(session_id)
        if not chunks:
            return []

        k = self.top_k if top_k is None else top_k
        if k is not None and k < 0:
            raise ValueError("top_k must be non-negative or None")
        query_terms = self._tokenize(query) or ["general"]

        # Tokenize every chunk.
        tokenized = [self._tokenize(text) for _, text, _, _ in chunks]

        # Compute IDF.
        n = len(chunks)
        df: Counter = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        idf = {
            term: math.log(
                1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5)
            )
            for term in set(query_terms)
        }
        avg_len = sum(len(t) for t in tokenized) / max(n, 1)

        scored: list[MemoryEntry] = []
        for pos, ((cid, text, turn, role), tokens) in enumerate(zip(chunks, tokenized)):
            score = self._bm25_score(tokens, query_terms, idf, avg_len)
            scored.append(MemoryEntry(id=cid, text=text, score=score, turn=turn, role=role, source_index=pos))

        scored.sort(key=lambda e: (-e.score, e.turn))
        return scored[:k]

    def clear(self, session_id: str) -> None:
        """Clear all memory for one session."""
        self._chunks.pop(session_id, None)

    def count(self, session_id: str) -> int:
        return len(self._chunks.get(session_id, []))

    def format_context_block(
        self,
        entries: list[MemoryEntry],
        max_chars: int = 2000,
        max_selected: Optional[int] = None,
    ) -> str:
        """Format entries as a text block for insertion into the LLM prompt.

        Selection takes the top-k entries by BM25 score (the input is already
        sorted in descending order), then presents them by ascending turn to
        preserve chronology. Fixed outer Track-B tags are added after
        packing and are not counted against the packing cap.
        """
        selected = []
        used = 0
        count_cap_binding = False
        for entry in entries:
            line = f"- (turn={entry.turn}) {entry.text}"
            separator = int(bool(selected))
            room = max_chars - used - separator
            if max_selected is not None and len(selected) >= max_selected:
                if len(line) <= room or (self.packing_mode == "truncate_last" and room > 50):
                    count_cap_binding = True
                    break
                if self.packing_mode == "whole_unit_skip":
                    continue
                break
            if len(line) > room:
                if self.packing_mode == "whole_unit_skip":
                    continue
                if self.packing_mode == "truncate_last" and room > 50:
                    selected.append((entry.source_order, line[:room], None, room))
                break
            selected.append((entry.source_order, line, entry.id, len(entry.text)))
            used += separator + len(line)

        selected.sort(key=lambda item: item[0])
        payload = "\n".join(line for _, line, _, _ in selected)
        context = self.header + payload + self.footer if selected else ""
        ids = [cid for _, _, cid, _ in selected if cid is not None]
        self.last_packing_stats = {
            "cap_chars": max_chars,
            "max_selected": max_selected,
            "selected_unit_count": len(selected),
            "count_cap_binding": count_cap_binding,
            "selected_payload_chars": sum(size for _, _, _, size in selected),
            "budgeted_chars": len(payload),
            "serialized_chars": len(context),
            "selected_entry_ids": ids,
            "selected_chunk_ids": ids,
            "selected_source_indices": [order[1] for order, _, _, _ in selected],
            "packing_mode": self.packing_mode,
        }
        return context
