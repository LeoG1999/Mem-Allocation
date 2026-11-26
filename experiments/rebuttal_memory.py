"""Query-blind memory views and budget accounting for the appendix controls."""

import json
import random
import re

from src.evaluation.longmemeval_runner import iter_history_turns, LongMemEvalRunner, session_occurrence_ids
from src.memory.dual_track import DualTrackMemory
from src.memory.track_b_rag import SemanticRAGTrack, TRACK_B_FOOTER, TRACK_B_HEADER, split_chunk_limit, wrapper_chars

FACT_PROMPT = """Extract durable facts stated in this conversation session. Return a JSON
array of triples of strings [entity, attribute, value]. Preserve dates, preferences
and revisions. Return [] when there are no supported facts.
Session:
"""
GRID = (0, 25, 33, 50, 67, 75, 100)


def thin_history(item, background_count, seed):
    ids = item["haystack_session_ids"]
    dates = item["haystack_dates"]
    sessions = item["haystack_sessions"]
    support = set(item["answer_session_ids"])
    if not len(sessions) == len(ids) == len(dates):
        raise ValueError("Invalid session IDs, dates or sessions")
    if not support.issubset(ids):
        raise ValueError("Missing supporting session")
    supporting = [i for i, sid in enumerate(ids) if sid in support]
    background = [i for i, sid in enumerate(ids) if sid not in support]
    random.Random(f"{seed}/{item['question_id']}").shuffle(background)
    keep = supporting + (background if background_count is None else background[:background_count])
    # Create a clean benchmark input and preserve date context without evidence flags.
    kept = sorted(
        keep,
        key=lambda i: (LongMemEvalRunner._parse_haystack_date(dates[i]), i),
    )
    result = {k: item[k] for k in ("question_id", "question_type", "question", "answer")}
    if "question_date" in item:
        result["question_date"] = item["question_date"]
    # Distinct occurrences can share a source ID; retain their individual dates and memory units.
    occurrences = session_occurrence_ids(ids)
    result["haystack_session_ids"] = [occurrences[i] for i in kept]
    result["haystack_dates"] = [dates[i] for i in kept]
    clean = []
    for i in kept:
        turns = [{k: msg[k] for k in ("role", "content")} for msg in sessions[i]]
        clean.append(turns)
    result["haystack_sessions"] = clean
    return result


def make_memory(sample, client, cache, chunk_chars):
    memory = DualTrackMemory(
        llm_client=client,
        summary_cache_dir=cache,
        chunk_size=chunk_chars // 4,
        chunk_overlap=chunk_chars // 32,
    )
    for turn, pair in enumerate(iter_history_turns(sample.history)):
        memory.add_turn("all", turn, pair["user"], pair["assistant"], real_session_id=pair["session_id"])
    memory.track_a.prepare_summaries()
    return memory


def extractive_text(text, length):
    # Rank sentences by shared vocabulary with the session; render in source order.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    terms = [set(SemanticRAGTrack._tokenize(s)) for s in sentences]
    scores = [sum(len(t & other) for other in terms) for t in terms]
    chosen = []
    size = 0
    for idx in sorted(range(len(sentences)), key=lambda i: -scores[i]):
        chosen.append(idx)
        size += len(sentences[idx]) + 1
        if size > length:
            break
    return " ".join(sentences[i] for i in sorted(chosen))[:length]


def alternative_views(memory, client):
    extracts = {}
    facts = SemanticRAGTrack()
    for index, sid in enumerate(memory.track_a._session_order):
        state = memory.track_a._states[sid]
        source = "\n".join(t for _, user, assistant in state.turns for t in (user, assistant))
        extracts[sid] = extractive_text(source, len(state.summary))
        response = client.chat(
            messages=[{"role": "user", "content": FACT_PROMPT + source}], temperature=0, max_tokens=512
        )
        triples = json.loads(response.content)
        if not isinstance(triples, list):
            raise ValueError("Expected a list of fact triples")
        for triple in triples:
            if (
                not isinstance(triple, list)
                or len(triple) != 3
                or not all(isinstance(x, str) for x in triple)
            ):
                raise ValueError("Invalid fact triple")
            # An entire triple is a retrieval unit, even when it exceeds the cap.
            entry = json.dumps(triple, ensure_ascii=False)
            bucket = facts._chunks.setdefault("all", [])
            bucket.append((str(len(bucket)), entry, index, "fact"))
    return extracts, facts


def ranked_views(memory, question, alternatives=None):
    track = memory.track_a
    order = {sid: i for i, sid in enumerate(track._session_order)}

    def sessions(texts):
        candidates = [sid for sid in order if texts[sid]]
        ranked = track._bm25_rank_sessions(candidates, question) if question else reversed(candidates)
        return [(order[sid], f"[Session {order[sid]+1}]\n{texts[sid]}") for sid in ranked]

    hits = memory.track_b.search("all", question)
    views = {
        "summary": sessions({sid: track.get_summary(sid) for sid in order}),
        "chunk": [(hit.source_order, f"- (turn={hit.turn}) {hit.text}") for hit in hits],
    }
    if alternatives is not None:
        extracts, facts = alternatives
        views["extract"] = sessions(extracts)
        views["fact"] = [
            (hit.source_order, hit.text) for hit in facts.search("all", question)
        ]
    return views


WRAPPERS = {
    "summary": ("[Episodic Memory]\n", "\n[/Episodic Memory]"),
    "extract": ("[Episodic Memory]\n", "\n[/Episodic Memory]"),
    "chunk": (TRACK_B_HEADER, TRACK_B_FOOTER),
    "fact": ("[Facts]\n", "\n[/Facts]"),
}


def pack(views, shares, budget, measure, *, character_budget=False):
    """Take whole units in BM25 order; check local shares and the joined block."""
    blocks = []
    chunk_wrapper_chars = 0
    allocated = 0
    for view, percent in shares:
        cap = budget * (allocated + percent) // 100 - budget * allocated // 100
        allocated += percent
        header, footer = WRAPPERS[view]
        exempt = len(header) + len(footer) if character_budget and view == "chunk" else 0
        if character_budget and blocks and view == "chunk":
            cap -= 2
        selected = []
        current = ""
        limit = split_chunk_limit(budget, token_budget=not character_budget) if view == "chunk" and 0 < percent < 100 else None
        for unit in views[view]:
            if limit is not None and len(selected) == limit:
                break
            trial = sorted(selected + [unit], key=lambda pair: pair[0])
            block = header + "\n".join(line for _, line in trial) + footer
            total = measure("\n\n".join(blocks + [block])) - chunk_wrapper_chars - exempt
            if measure(block) - exempt <= cap and total <= budget:
                selected = trial
                current = block
        if current:
            blocks.append(current)
            chunk_wrapper_chars += exempt
    context = "\n\n".join(blocks)
    return context, measure(context) - chunk_wrapper_chars


def char_context(memory, question, budget, ratio):
    return memory.get_context("all", query=question, budget_chars=budget, track_a_ratio_override=ratio / 100)


def realized_chars(context):
    return len(context) - wrapper_chars(context)
