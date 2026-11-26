"""Dual-Track combines session summaries and raw-history chunks within a
character budget.

It supports the main, allocation, masking, coverage, and extended-budget
experiments in the paper.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.memory.track_a import EpisodicTrack
from src.memory.track_b_rag import SemanticRAGTrack, TRACK_B_LABEL, split_chunk_limit
from src.models.llm_client import LLMClient

DEFAULT_BUDGET = 4000  # Default character budget when no limit is provided


class QueryType(Enum):
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    CONFLICT = "conflict"
    BALANCED = "balanced"


TEMPORAL_KEYWORDS = [
    "之前", "之后", "先", "后", "then", "before", "after",
    "when did", "what happened", "previously", "earlier",
    "sequence", "order", "first", "last", "最近", "上次",
    "为什么", "why", "因为", "because", "导致", "caused",
    # cross-session comparison / multi-session signals
    "compared", "comparison", "more expensive", "less expensive",
    "how much more", "how much less", "difference between",
    "over time", "across", "throughout", "history of",
    "total", "overall", "cumulative", "combined",
    "each session", "multiple", "several times",
]

CONFLICT_KEYWORDS = [
    "changed", "updated", "修改", "更新", "现在", "now",
    "still", "还是", "anymore", "不再", "new", "latest",
    "conflict", "矛盾", "之前说", "但是",
    "current", "currently", "at this point", "most recent",
    "replaced", "switched", "instead",
]


@dataclass
class RouterDecision:
    query_type: QueryType
    track_a_weight: float
    track_b_weight: float
    track_a_chars: int
    track_b_chars: int


class MemoryRouter:
    def route(self, query: str, budget_chars: int) -> RouterDecision:
        q = query.lower()
        t_score = sum(1 for kw in TEMPORAL_KEYWORDS if kw in q)
        c_score = sum(1 for kw in CONFLICT_KEYWORDS if kw in q)

        if c_score >= 2:
            qt, a_w, b_w = QueryType.CONFLICT, 0.5, 0.5
        elif t_score >= 2:
            qt, a_w, b_w = QueryType.TEMPORAL, 0.7, 0.3
        elif t_score == 1:
            qt, a_w, b_w = QueryType.BALANCED, 0.5, 0.5
        elif t_score == 0 and c_score == 0:
            qt, a_w, b_w = QueryType.SEMANTIC, 0.5, 0.5  # equal split by default
        else:
            qt, a_w, b_w = QueryType.BALANCED, 0.5, 0.5

        return RouterDecision(
            query_type=qt,
            track_a_weight=a_w,
            track_b_weight=b_w,
            track_a_chars=int(budget_chars * a_w),
            track_b_chars=int(budget_chars * b_w),
        )


class DualTrackMemory:
    """Allocate the Track-A/B character budget by a static ratio or router decision."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        persist_dir: str = "data/cache/dual_track",
        use_llm_memo: bool = True,
        router: Optional[MemoryRouter] = None,
        enable_track_a: bool = True,
        enable_track_b: bool = True,
        use_router: bool = False,
        mask_track_a: bool = False,
        mask_track_b: bool = False,
        max_memos: int = 20,
        memo_in_context: int = 5,
        top_k: Optional[int] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        track_a_ratio: float = 0.5,
        summary_cache_dir: Optional[str] = None,
        track_b_packing_mode: str = "whole_unit_skip",
        max_chars_per_session: Optional[int] = None,
        track_b_label: str = TRACK_B_LABEL,
        five_chunk_sweep: bool = True,
    ):
        self.five_chunk_sweep = five_chunk_sweep
        self.enable_track_a = enable_track_a
        self.enable_track_b = enable_track_b
        self.use_router = use_router
        self.mask_track_a = mask_track_a
        self.mask_track_b = mask_track_b
        self.track_a_ratio = max(0.0, min(1.0, track_a_ratio))

        self.track_a = EpisodicTrack(
            llm_client=llm_client,
            use_llm_summary=use_llm_memo,
            memo_in_context=memo_in_context,
            max_chars_per_session=max_chars_per_session,
            summary_cache_dir=summary_cache_dir,
        ) if enable_track_a else None

        self.track_b = SemanticRAGTrack(
            persist_dir=persist_dir,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            packing_mode=track_b_packing_mode,
            label=track_b_label,
        ) if enable_track_b else None

        self.router = router or MemoryRouter()

    def add_turn(
        self,
        session_id: str,
        turn: int,
        user_content: str,
        assistant_content: str,
        real_session_id: Optional[str] = None,
    ) -> None:
        # Track A uses the real session ID to preserve cross-session chronology.
        # Track B uses the supplied ID, normally MERGED_SESSION_ID, for global retrieval.
        track_a_sid = real_session_id if real_session_id else session_id
        if self.track_a:
            self.track_a.add_turn(track_a_sid, turn, user_content, assistant_content)
        if self.track_b:
            self.track_b.add(
                session_id,
                messages=[
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                turn=turn,
            )

    def get_context(
        self,
        session_id: str,
        query: str = "",
        budget_chars: Optional[int] = None,
        track_a_ratio_override: Optional[float] = None,
    ) -> str:
        ctx, _ = self.get_context_with_attribution(
            session_id, query, budget_chars, track_a_ratio_override
        )
        return ctx

    def get_context_with_attribution(
        self,
        session_id: str,
        query: str = "",
        budget_chars: Optional[int] = None,
        track_a_ratio_override: Optional[float] = None,
    ) -> tuple[str, dict]:
        """
        Return (context_str, attribution_dict).

        attribution_dict supports the orthogonality analysis:
        {
            "query_type": "temporal" | "semantic" | ...,
            "track_a_chars": int,   # Characters actually emitted by Track A
            "track_b_chars": int,   # Characters actually emitted by Track B
            "track_a_context": str, # Raw Track-A output
            "track_b_context": str, # Raw Track-B output
        }
        """
        budget = DEFAULT_BUDGET if budget_chars is None else budget_chars
        # Per-call ratio override takes priority over instance default
        ratio = track_a_ratio_override if track_a_ratio_override is not None else self.track_a_ratio

        if self.use_router:
            decision = self.router.route(query or "", budget)
        else:
            # Static split with configurable ratio (default 50/50)
            a_chars = int(budget * ratio)
            decision = RouterDecision(
                query_type=QueryType.BALANCED,
                track_a_weight=ratio,
                track_b_weight=1.0 - ratio,
                track_a_chars=a_chars,
                track_b_chars=budget - a_chars,
            )

        track_a_ctx = ""
        track_b_ctx = ""

        if self.track_a and self.enable_track_a:
            # Track A receives the full budget when Track B is disabled.
            a_budget = budget if not self.enable_track_b else decision.track_a_chars
            track_a_ctx = self.track_a.get_context(
                session_id, query=query or "", budget_chars=a_budget
            )

        if self.track_b and self.enable_track_b:
            hits = self.track_b.search(session_id, query=query or "general context") if decision.track_b_chars else []
            # Reserve two newline characters from Track B when both tracks are nonempty.
            track_b_payload_cap = max(
                0,
                decision.track_b_chars - (2 if track_a_ctx else 0),
            )
            track_b_ctx = self.track_b.format_context_block(
                hits, max_chars=track_b_payload_cap,
                max_selected=(split_chunk_limit(budget) if self.five_chunk_sweep and self.enable_track_a
                              and 0 < decision.track_a_weight < 1 else None),
            )

        # Table 13 writes full history to both tracks and masks only at read time.
        masked_a = "" if self.mask_track_a else track_a_ctx
        masked_b = "" if self.mask_track_b else track_b_ctx

        parts = [p for p in [masked_a, masked_b] if p]
        combined = "\n\n".join(parts)

        attribution = {
            "query_type": decision.query_type.value,
            "use_router": self.use_router,
            "mask_track_a": self.mask_track_a,
            "mask_track_b": self.mask_track_b,
            "track_a_chars": len(masked_a),
            "track_b_chars": len(masked_b),
            "track_a_context": masked_a,
            "track_b_context": masked_b,
            "track_a_cap_chars": decision.track_a_chars,
            "track_b_cap_chars": decision.track_b_chars,
            "track_a_packing": (
                dict(getattr(self.track_a, "last_packing_stats", {}))
                if self.track_a else {}
            ),
            "track_b_packing": (
                dict(getattr(self.track_b, "last_packing_stats", {}))
                if self.track_b else {}
            ),
        }

        return combined, attribution

    def clear(self, session_id: str) -> None:
        if self.track_a:
            self.track_a.clear(session_id)
        if self.track_b:
            self.track_b.clear(session_id)

    def stats(self, session_id: str) -> dict:
        return {
            "track_a_sessions": len(self.track_a._session_order) if self.track_a else 0,
            "track_a_summary_len": len(self.track_a.get_summary(session_id)) if self.track_a else 0,
            "track_b_chunks": self.track_b.count(session_id) if self.track_b else 0,
        }
