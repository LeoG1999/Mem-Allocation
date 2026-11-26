"""Matched dense-retrieval strategies used in Table 3."""

from __future__ import annotations

from typing import Optional

from src.memory.dual_track import DualTrackMemory
from src.memory.track_a import EpisodicTrack
from src.memory.track_b_rag import (
    MemoryEntry,
    SemanticRAGTrack,
)
from src.memory.baselines.strategies import BM25ChunkTrackBStrategy


class _DenseEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            [query] + documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(vectors[0] @ vector) for vector in vectors[1:]]


class DenseEpisodicTrack(EpisodicTrack):
    """Rank sessions with MiniLM while leaving summary construction unchanged."""

    def __init__(self, *args, model_name: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dense = _DenseEncoder(model_name)

    def _bm25_rank_sessions(self, session_order: list[str], query: str) -> list[str]:
        if not query or not session_order:
            return session_order
        documents = []
        for sid in session_order:
            state = self._states.get(sid)
            documents.append(
                " ".join(f"{user} {assistant}" for _, user, assistant in state.turns)
                if state else ""
            )
        scores = self._dense.scores(query, documents)
        return [sid for _, sid in sorted(zip(scores, session_order), reverse=True)]


class DenseSemanticRAGTrack(SemanticRAGTrack):
    """Rank Track-B chunks with MiniLM while leaving chunking and packing unchanged."""

    def __init__(self, *args, model_name: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dense = _DenseEncoder(model_name)

    def search(
        self,
        session_id: str | list[str],
        query: str,
        top_k: Optional[int] = None,
    ) -> list[MemoryEntry]:
        chunks = self.chunks_for(session_id)
        if not chunks:
            return []
        scores = self._dense.scores(query or "general context", [row[1] for row in chunks])
        entries = [
            MemoryEntry(id=cid, text=text, score=score, turn=turn, role=role, source_index=pos)
            for pos, ((cid, text, turn, role), score) in enumerate(zip(chunks, scores))
        ]
        entries.sort(key=lambda row: (-row.score, row.turn))
        return entries[: self.top_k if top_k is None else top_k]


class DenseDualTrackMemory(DualTrackMemory):
    """Replace the Track-A and Track-B rankers together."""

    def __init__(
        self,
        *,
        llm_client,
        model_name: str,
        summary_cache_dir: str,
        packing_mode: str,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            use_router=False,
            track_a_ratio=0.5,
            top_k=top_k,
            summary_cache_dir=summary_cache_dir,
            track_b_packing_mode=packing_mode,
        )
        self.track_a = DenseEpisodicTrack(
            llm_client=llm_client,
            use_llm_summary=True,
            summary_cache_dir=summary_cache_dir,
            model_name=model_name,
        )
        self.track_b = DenseSemanticRAGTrack(
            top_k=top_k,
            chunk_size=256,
            chunk_overlap=32,
            packing_mode=packing_mode,
            model_name=model_name,
        )


class DenseDualStrategy:
    name = "dense_split"

    def __init__(self, **kwargs) -> None:
        self._mem = DenseDualTrackMemory(**kwargs)

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


class DenseChunkTrackBStrategy(BM25ChunkTrackBStrategy):
    """Full-budget dense Track-B control for Table 3."""

    name = "dense_trackb"

    track_type = DenseSemanticRAGTrack
