"""Recompute Table 6 chunk-only utilization and the Section 3.2 count-cap sensitivity."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.longmemeval_runner import LongMemEvalRunner, iter_history_turns
from src.memory.baselines.strategies import BM25ChunkTrackBStrategy, MERGED_SESSION_ID
from src.memory.track_b_rag import split_chunk_limit


def audit(samples, budgets):
    totals = {b: Counter() for b in budgets}
    for sample in samples:
        strategy = BM25ChunkTrackBStrategy()
        for turn, pair in enumerate(iter_history_turns(sample.history)):
            strategy.add_turn(pair["session_id"], turn, pair["user"], pair["assistant"])
        sid = MERGED_SESSION_ID if len(sample.session_ids) > 1 else sample.session_ids[0]
        hits = strategy.track.search(strategy._material_session_ids(sid), sample.question or "general context")
        for budget, counts in totals.items():
            pack = strategy.track.format_context_block
            pack(hits, budget)
            stats = strategy.track.last_packing_stats
            counts["chunks"] += stats["selected_unit_count"]
            counts["chars"] += stats["budgeted_chars"]
            # With an empty Track A, B gets its full share. Otherwise reserve the blank line.
            changes = []
            for separator in (0, 2):
                share = max(0, budget - int(budget * 0.5) - separator)
                limit = split_chunk_limit(budget)
                changes.append(pack(hits, share) != pack(hits, share, max_selected=limit))
            counts["always_changes"] += all(changes)
            counts["possibly_changes"] += any(changes)
            counts["n"] += 1
    result = {}
    for budget, counts in totals.items():
        n = counts["n"]
        if not n:
            raise ValueError("No benchmark questions selected")
        result[budget] = dict(n=n, concentration_chunks_per_item=counts["chunks"] / n,
                              concentration_injected_chars=counts["chars"] / n,
                              five_chunk_changed_items_min=counts["always_changes"],
                              five_chunk_changed_items_max=counts["possibly_changes"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/longmemeval/longmemeval_s.json")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1000, 2000, 4000, 6000])
    args = parser.parse_args()
    if min(args.budgets) <= 0:
        parser.error("Budgets must be positive")
    print(json.dumps(audit(LongMemEvalRunner(args.data)._load_data(), args.budgets), indent=2))
