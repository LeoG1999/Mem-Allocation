"""
Experiment utilities:
- load shared configuration
- construct strategies
- save JSON and JSONL consistently
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import yaml

from src.memory.baselines.strategies import (
    BM25ChunkStrategy,
    BM25ChunkTrackBStrategy,
    BM25SessionStrategy,
    BaseMemoryStrategy,
    DualTrackMaskAStrategy,
    DualTrackMaskBStrategy,
    DualTrackStrategy,
    EmbeddingChunkStrategy,
    EmbeddingSessionStrategy,
    FullContextStrategy,
    RecentSessionsStrategy,
    SessionSummaryStrategy,
    TruncationStrategy,
)
from src.memory.dual_track import DualTrackMemory
from src.memory.track_b_rag import TRACK_B_LABEL


def _attach_factory(strategy: BaseMemoryStrategy, factory) -> BaseMemoryStrategy:
    strategy._factory = factory
    return strategy


class DualTrackStrategyAdapter(BaseMemoryStrategy):
    name = "dual_track"

    def __init__(
        self,
        llm_client=None,
        persist_dir="data/cache/dual_track",
        use_llm_memo: bool = True,
        max_memos: int = 20,
        memo_in_context: int = 5,
        top_k: Optional[int] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        packing_mode: str = "whole_unit_skip",
        summary_cache_dir: Optional[str] = None,
        max_chars_per_session: Optional[int] = None,
        track_b_label: str = TRACK_B_LABEL,
        five_chunk_sweep: bool = True,
    ):
        self._mem = DualTrackMemory(
            llm_client=llm_client,
            persist_dir=persist_dir,
            use_llm_memo=use_llm_memo,
            max_memos=max_memos,
            memo_in_context=memo_in_context,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            track_b_packing_mode=packing_mode,
            max_chars_per_session=max_chars_per_session,
            track_b_label=track_b_label,
            five_chunk_sweep=five_chunk_sweep,
            summary_cache_dir=summary_cache_dir,
        )

    def add_turn(self, session_id, turn, user_content, assistant_content, real_session_id=None):
        self._mem.add_turn(session_id, turn, user_content, assistant_content, real_session_id=real_session_id)

    def get_context(self, session_id, query="", budget_chars=None):
        return self._mem.get_context(session_id, query, budget_chars)

    def clear(self, session_id):
        self._mem.clear(session_id)


def load_experiment_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        config = yaml.safe_load(f)
    if "base_config" not in config:
        return config
    base_path = os.path.join(os.path.dirname(path), config["base_config"])
    with open(base_path) as f:
        base = yaml.safe_load(f)
    for name, settings in config["strategy_overrides"].items():
        base["strategies"][name].update(settings)
    return base


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path: str, payload: Any) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
    return path


def save_jsonl(path: str, rows: List[dict]) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    return path


def get_strategy_group(
    config: Optional[Dict[str, Any]],
    key: str,
    fallback: Optional[List[str]] = None,
) -> List[str]:
    cfg = config or {}
    evaluation_cfg = cfg.get("evaluation", {})
    names = evaluation_cfg.get(key)
    if names:
        return list(names)
    return list(fallback or [])


def build_strategies_from_config(
    llm_client,
    config: Optional[Dict[str, Any]] = None,
    include_full_context: bool = True,
    persist_root: str = "data/cache/default",
) -> List[BaseMemoryStrategy]:
    cfg = config or {}
    strat_cfg = cfg.get("strategies", {})
    recent_cfg = strat_cfg.get("recent_sessions", strat_cfg.get("sliding_window", {}))
    summary_cfg = strat_cfg.get("session_summary", {})
    rag_cfg = strat_cfg.get("bm25_session", strat_cfg.get("session_rag", strat_cfg.get("rag_only", {})))
    dual_cfg = strat_cfg.get("dual_track", {})

    strategies: List[BaseMemoryStrategy] = []
    if include_full_context:
        strategies.append(_attach_factory(FullContextStrategy(), lambda: FullContextStrategy()))
    strategies.append(_attach_factory(TruncationStrategy(), lambda: TruncationStrategy()))
    strategies.append(
        _attach_factory(
            RecentSessionsStrategy(
                window_size=recent_cfg.get("window_size"),
                max_chars_per_turn=recent_cfg.get("max_chars_per_turn"),
                max_turns_per_session=recent_cfg.get("max_turns_per_session"),
            ),
            lambda: RecentSessionsStrategy(
                window_size=recent_cfg.get("window_size"),
                max_chars_per_turn=recent_cfg.get("max_chars_per_turn"),
                max_turns_per_session=recent_cfg.get("max_turns_per_session"),
            ),
        )
    )
    ss_cache = os.path.join(persist_root, "summary_cache")
    strategies.append(
        _attach_factory(
            SessionSummaryStrategy(
                llm_client=llm_client,
                recent_session_count=summary_cfg.get("recent_session_count", 8),
                max_chars_per_turn=summary_cfg.get("max_chars_per_turn"),
                max_turns_per_session=summary_cfg.get("max_turns_per_session"),
                use_llm_summary=summary_cfg.get("use_llm_summary", True),
                llm_input_char_limit=summary_cfg.get("llm_input_char_limit"),
                max_tokens_summary=summary_cfg.get("max_tokens_summary", 300),
                query_aware=summary_cfg.get("query_aware", True),
                summary_cache_dir=ss_cache,
            ),
            lambda: SessionSummaryStrategy(
                llm_client=llm_client,
                recent_session_count=summary_cfg.get("recent_session_count", 8),
                max_chars_per_turn=summary_cfg.get("max_chars_per_turn"),
                max_turns_per_session=summary_cfg.get("max_turns_per_session"),
                use_llm_summary=summary_cfg.get("use_llm_summary", True),
                llm_input_char_limit=summary_cfg.get("llm_input_char_limit"),
                max_tokens_summary=summary_cfg.get("max_tokens_summary", 300),
                query_aware=summary_cfg.get("query_aware", True),
                summary_cache_dir=ss_cache,
            ),
        )
    )
    strategies.append(
        _attach_factory(
            BM25SessionStrategy(
                top_k=rag_cfg.get("top_k"),
                max_chars_per_turn=rag_cfg.get("max_chars_per_turn"),
                recency_bias=rag_cfg.get("recency_bias", 0.0),
                k1=rag_cfg.get("k1", 1.5),
                b=rag_cfg.get("b", 0.75),
            ),
            lambda: BM25SessionStrategy(
                top_k=rag_cfg.get("top_k"),
                max_chars_per_turn=rag_cfg.get("max_chars_per_turn"),
                recency_bias=rag_cfg.get("recency_bias", 0.0),
                k1=rag_cfg.get("k1", 1.5),
                b=rag_cfg.get("b", 0.75),
            ),
        )
    )
    emb_cfg = strat_cfg.get("embedding_session", {})
    strategies.append(
        _attach_factory(
            EmbeddingSessionStrategy(
                model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
                top_k=emb_cfg.get("top_k", 8),
                max_chars_per_turn=emb_cfg.get("max_chars_per_turn", 220),
                recency_bias=emb_cfg.get("recency_bias", 0.02),
            ),
            lambda: EmbeddingSessionStrategy(
                model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
                top_k=emb_cfg.get("top_k", 8),
                max_chars_per_turn=emb_cfg.get("max_chars_per_turn", 220),
                recency_bias=emb_cfg.get("recency_bias", 0.02),
            ),
        )
    )
    emb_chunk_cfg = strat_cfg.get("embedding_chunk", {})
    strategies.append(
        _attach_factory(
            EmbeddingChunkStrategy(
                model_name=emb_chunk_cfg.get("model_name", "all-MiniLM-L6-v2"),
                top_k=emb_chunk_cfg.get("top_k", 8),
                chunk_size=emb_chunk_cfg.get("chunk_size", 64),
                chunk_overlap=emb_chunk_cfg.get("chunk_overlap", 16),
                recency_bias=emb_chunk_cfg.get("recency_bias", 0.02),
            ),
            lambda: EmbeddingChunkStrategy(
                model_name=emb_chunk_cfg.get("model_name", "all-MiniLM-L6-v2"),
                top_k=emb_chunk_cfg.get("top_k", 8),
                chunk_size=emb_chunk_cfg.get("chunk_size", 64),
                chunk_overlap=emb_chunk_cfg.get("chunk_overlap", 16),
                recency_bias=emb_chunk_cfg.get("recency_bias", 0.02),
            ),
        )
    )
    bm25_chunk_cfg = strat_cfg.get("bm25_chunk", {})
    strategies.append(
        _attach_factory(
            BM25ChunkStrategy(
                top_k=bm25_chunk_cfg.get("top_k", 8),
                chunk_size=bm25_chunk_cfg.get("chunk_size", 64),
                chunk_overlap=bm25_chunk_cfg.get("chunk_overlap", 16),
                recency_bias=bm25_chunk_cfg.get("recency_bias", 0.02),
                k1=bm25_chunk_cfg.get("k1", 1.5),
                b=bm25_chunk_cfg.get("b", 0.75),
            ),
            lambda: BM25ChunkStrategy(
                top_k=bm25_chunk_cfg.get("top_k", 8),
                chunk_size=bm25_chunk_cfg.get("chunk_size", 64),
                chunk_overlap=bm25_chunk_cfg.get("chunk_overlap", 16),
                recency_bias=bm25_chunk_cfg.get("recency_bias", 0.02),
                k1=bm25_chunk_cfg.get("k1", 1.5),
                b=bm25_chunk_cfg.get("b", 0.75),
            ),
        )
    )
    # bm25_chunk_trackb: full-budget allocation control using Track B's exact pipeline
    trackb_cfg = strat_cfg.get("bm25_chunk_trackb", {})
    strategies.append(
        _attach_factory(
            BM25ChunkTrackBStrategy(
                chunk_size_chars=trackb_cfg.get("chunk_size_chars", 1024),
                chunk_overlap_chars=trackb_cfg.get("chunk_overlap_chars", 128),
                k1=trackb_cfg.get("k1", 1.5),
                top_k=trackb_cfg.get("top_k"),
                label=trackb_cfg.get("track_b_label", dual_cfg.get("track_b_label", TRACK_B_LABEL)),
                b=trackb_cfg.get("b", 0.75),
                packing_mode=trackb_cfg.get("packing_mode", "whole_unit_skip"),
            ),
            lambda: BM25ChunkTrackBStrategy(
                chunk_size_chars=trackb_cfg.get("chunk_size_chars", 1024),
                chunk_overlap_chars=trackb_cfg.get("chunk_overlap_chars", 128),
                k1=trackb_cfg.get("k1", 1.5),
                top_k=trackb_cfg.get("top_k"),
                label=trackb_cfg.get("track_b_label", dual_cfg.get("track_b_label", TRACK_B_LABEL)),
                b=trackb_cfg.get("b", 0.75),
                packing_mode=trackb_cfg.get("packing_mode", "whole_unit_skip"),
            ),
        )
    )
    summary_cache = os.path.join(persist_root, "summary_cache")
    strategies.append(
        _attach_factory(
            DualTrackStrategyAdapter(
                llm_client=llm_client,
                persist_dir=os.path.join(persist_root, f"dual_track_{uuid.uuid4().hex}"),
                use_llm_memo=dual_cfg.get("use_llm_memo", True),
                max_memos=dual_cfg.get("max_memos", 20),
                memo_in_context=dual_cfg.get("memo_in_context", 5),
                top_k=dual_cfg.get("top_k"),
                max_chars_per_session=summary_cfg.get("llm_input_char_limit"),
                track_b_label=dual_cfg.get("track_b_label", TRACK_B_LABEL),
                five_chunk_sweep=dual_cfg.get("five_chunk_sweep", True),
                chunk_size=dual_cfg.get("chunk_size", 256),
                chunk_overlap=dual_cfg.get("chunk_overlap", 32),
                packing_mode=dual_cfg.get("packing_mode", "whole_unit_skip"),
                summary_cache_dir=summary_cache,
            ),
            lambda: DualTrackStrategyAdapter(
                llm_client=llm_client,
                persist_dir=os.path.join(persist_root, f"dual_track_{uuid.uuid4().hex}"),
                use_llm_memo=dual_cfg.get("use_llm_memo", True),
                max_memos=dual_cfg.get("max_memos", 20),
                memo_in_context=dual_cfg.get("memo_in_context", 5),
                top_k=dual_cfg.get("top_k"),
                max_chars_per_session=summary_cfg.get("llm_input_char_limit"),
                track_b_label=dual_cfg.get("track_b_label", TRACK_B_LABEL),
                five_chunk_sweep=dual_cfg.get("five_chunk_sweep", True),
                chunk_size=dual_cfg.get("chunk_size", 256),
                chunk_overlap=dual_cfg.get("chunk_overlap", 32),
                packing_mode=dual_cfg.get("packing_mode", "whole_unit_skip"),
                summary_cache_dir=summary_cache,
            ),
        )
    )
    return strategies
