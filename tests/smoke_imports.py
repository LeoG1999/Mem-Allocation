#!/usr/bin/env python3
"""Lightweight import smoke test that does not call external APIs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.longmemeval_runner import LongMemEvalRunner  # noqa: F401
from src.evaluation.locomo_runner import LoCoMoRunner  # noqa: F401
from src.memory.dual_track import DualTrackMemory  # noqa: F401
from src.memory.dense_matched import DenseDualStrategy, DenseChunkTrackBStrategy  # noqa: F401
from src.memory.baselines.strategies import (  # noqa: F401
    BM25ChunkStrategy,
    BM25ChunkTrackBStrategy,
    BM25SessionStrategy,
    EmbeddingChunkStrategy,
    RecentSessionsStrategy,
    SessionSummaryStrategy,
)
from src.models.llm_client import LLMClient  # noqa: F401

print("smoke_imports: OK")
