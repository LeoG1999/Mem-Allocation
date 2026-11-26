import json
from types import SimpleNamespace

from src.experiment_utils import build_strategies_from_config, load_experiment_config
from src.memory.track_b_rag import payload_chars, wrapper_chars

from experiments.exp_paper_controls import (
    AllocationDualStrategy,
    condition_spec,
    parse_route_label,
)
from src.evaluation.longmemeval_runner import LongMemEvalRunner
from src.memory.baselines.strategies import BM25ChunkTrackBStrategy
from src.memory.dual_track import DualTrackMemory
from src.memory.track_b_rag import TRACK_B_FOOTER, TRACK_B_HEADER
from src.memory.dense_matched import _DenseEncoder, DenseChunkTrackBStrategy, DenseDualStrategy
from src.models.llm_client import LLMResponse


class DeterministicTestLLM:
    model = "test-model"

    def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "memory summary" in prompt:
            content = "- Alice moved to Berlin in 2024."
        else:
            content = "Berlin"
        return LLMResponse(
            content=content,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=2,
            model=self.model,
        )


def test_route_parser_is_strict_and_uses_mixed_as_default():
    assert parse_route_label("EVIDENCE") == ("EVIDENCE", True)
    assert parse_route_label("state") == ("STATE", True)
    assert parse_route_label("Label: state") == ("MIXED", False)
    assert parse_route_label("EVIDENCE or MIXED") == ("EVIDENCE", True)
    assert parse_route_label("unclear") == ("MIXED", False)


def test_whole_unit_length_matching_never_exceeds_serialized_target():
    strategy = BM25ChunkTrackBStrategy(
        chunk_size_chars=64,
        chunk_overlap_chars=8,
        packing_mode="whole_unit_skip",
        budget_includes_wrapper=True,
    )
    for turn in range(4):
        strategy.add_turn(
            "s1", turn,
            f"Alice discussed Berlin travel plans number {turn} in detail.",
            f"The confirmed destination was Berlin for plan {turn}.",
        )
    context = strategy.get_context("s1", query="Where did Alice plan to travel?", budget_chars=240)
    assert len(context) <= 240
    assert strategy.last_packing_stats["packing_mode"] == "whole_unit_skip"
    assert strategy.last_packing_stats["selected_unit_count"] >= 1


def test_runner_persists_context_and_direct_packing_stats(tmp_path):
    data_path = tmp_path / "tiny.json"
    data_path.write_text(json.dumps([{
        "question_id": "q1",
        "question": "Where did Alice move?",
        "answer": "Berlin",
        "type": "single-session-user",
        "haystack_session_ids": ["session_1"],
        "haystack_dates": ["2024/01/01 (Mon) 10:00"],
        "haystack_sessions": [[
            {"role": "user", "content": "I moved to Berlin in 2024."},
            {"role": "assistant", "content": "I will remember that."},
        ]],
    }]))
    llm = DeterministicTestLLM()
    spec = condition_spec("split256")
    strategy = AllocationDualStrategy(
        name="split256",
        llm_client=llm,
        spec=spec,
        summary_cache_dir=str(tmp_path / "summary-cache"),
        use_llm_summary=False,
        packing_mode="whole_unit_skip",
    )
    report = LongMemEvalRunner(str(data_path), judge_llm=None).evaluate(
        strategy=strategy,
        answer_llm=llm,
        budget_chars=1000,
        judge_mode="string",
        store_context=True,
    )
    row = report.samples[0]
    assert row.is_correct
    assert row.memory_context
    assert row.memory_context_len == len(row.memory_context)
    assert row.attribution["track_a_cap_chars"] == 500
    assert row.attribution["track_b_cap_chars"] == 500
    assert row.attribution["track_a_packing"]["selected_unit_count"] == 1
    assert row.attribution["track_b_packing"]["selected_unit_count"] >= 1
    assert row.attribution["track_b_packing"]["packing_mode"] == "whole_unit_skip"


def test_dense_matched_strategies_run_on_fixture(tmp_path):
    data_path = tmp_path / "tiny_dense.json"
    data_path.write_text(json.dumps([{
        "question_id": "q1",
        "question": "Where did Alice move?",
        "answer": "Berlin",
        "type": "single-session-user",
        "haystack_session_ids": ["session_1"],
        "haystack_dates": ["2024/01/01 (Mon) 10:00"],
        "haystack_sessions": [[
            {"role": "user", "content": "I moved to Berlin in 2024."},
            {"role": "assistant", "content": "I will remember that."},
        ]],
    }]))
    original = _DenseEncoder.scores
    _DenseEncoder.scores = lambda self, query, documents: [float(len(doc)) for doc in documents]
    try:
        llm = DeterministicTestLLM()
        strategies = [
            DenseDualStrategy(
                llm_client=llm,
                model_name="fixture",
                summary_cache_dir=str(tmp_path / "dense-cache"),
                packing_mode="whole_unit_skip",
            ),
            DenseChunkTrackBStrategy(
                model_name="fixture",
                chunk_size_chars=1024,
                chunk_overlap_chars=128,
                packing_mode="whole_unit_skip",
            ),
        ]
        for strategy in strategies:
            report = LongMemEvalRunner(str(data_path)).evaluate(
                strategy=strategy,
                answer_llm=llm,
                budget_chars=1000,
                judge_mode="string",
            )
            assert report.samples[0].is_correct
    finally:
        _DenseEncoder.scores = original


def test_matched_track_b_uses_identical_chunks_and_fixed_outer_overhead(tmp_path):
    llm = DeterministicTestLLM()
    dual = DualTrackMemory(
        llm_client=llm,
        use_router=False,
        track_a_ratio=0.5,
        summary_cache_dir=str(tmp_path / "matched-cache"),
        track_b_packing_mode="whole_unit_skip",
    )
    concentration = BM25ChunkTrackBStrategy(packing_mode="whole_unit_skip")
    turns = [
        (
            "Alice described a Berlin itinerary with museum reservations and train times. " * 12,
            "The assistant confirmed the Berlin itinerary and the reservation details. " * 10,
        ),
        (
            "Alice later kept Berlin as the destination and changed only the departure time. " * 8,
            "The latest departure time was recorded without changing the destination. " * 8,
        ),
    ]
    for turn, (user, assistant) in enumerate(turns):
        dual.add_turn("merged", turn, user, assistant, real_session_id="session-1")
        concentration.add_turn("session-1", turn, user, assistant)

    dual_chunks = [text for _, text, _, _ in dual.track_b._chunks["merged"]]
    concentration_chunks = [text for _, text, _, _ in concentration.track.chunks_for("session-1")]
    assert dual_chunks == concentration_chunks

    context, attribution = dual.get_context_with_attribution(
        "merged", query="What was Alice's destination?", budget_chars=4000
    )
    track_b = attribution["track_b_context"]
    assert track_b.startswith(TRACK_B_HEADER)
    assert track_b.endswith(TRACK_B_FOOTER)
    assert (TRACK_B_HEADER, TRACK_B_FOOTER) == ("[Semantic Memory]\n", "\n[/Semantic Memory]")
    assert all(line.startswith("- (turn=") for line in track_b.splitlines()[1:-1])
    assert len(context) <= 4000 + len(TRACK_B_HEADER) + len(TRACK_B_FOOTER)
    split_stats = attribution["track_b_packing"]
    cap = split_stats["cap_chars"]
    assert concentration.get_context("session-1", "What was Alice's destination?", cap) == track_b
    assert split_stats["selected_source_indices"] == concentration.last_packing_stats["selected_source_indices"]
    assert split_stats["max_selected"] == 5
    assert concentration.last_packing_stats["max_selected"] is None


def test_candidate_depth_and_summary_inputs(tmp_path):
    class RecordingClient:
        model = "test"
        api_model = "test-v1"

        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages[0]["content"], kwargs))
            return SimpleNamespace(content="- Summary.")

    def strategies(cfg):
        return {s.name: s._factory() for s in build_strategies_from_config(
            client, cfg, persist_root=str(tmp_path / "protocol")
        ) if s.name in {"dual_track", "bm25_chunk_trackb", "session_summary"}}

    client = RecordingClient()
    cfg = load_experiment_config("configs/experiment_config.yaml")
    default = strategies(cfg)
    split = default["dual_track"]._mem
    control = default["bm25_chunk_trackb"]

    for turn in range(6):
        # The five highest-ranked units are too large for this cap.
        text = "needle " * 35 if turn < 5 else "short answer"
        split.add_turn("s", turn, text, "")
        control.add_turn("s", turn, text, "")
    exact = "- (turn=5) [user] short answer"
    cap = len(exact)
    half = split.get_context_with_attribution("s", "needle", cap * 2)[1]
    assert half["track_b_context"] == TRACK_B_HEADER + exact + TRACK_B_FOOTER
    assert control.get_context("s", "needle", cap) == half["track_b_context"]
    assert split.get_context("s", "needle", cap, 0) == half["track_b_context"]
    assert split.get_context("s", "needle", 0) == ""
    assert split.get_context("s", "needle", 4000, 1).endswith("[/Episodic Memory]")
    assert wrapper_chars(half["track_b_context"]) == 37
    other = "[Chunk Retrieval (BM25)]\n" + exact + "\n[/Chunk Retrieval (BM25)]"
    assert wrapper_chars(other) == 51
    assert payload_chars({}, other) == cap
    assert payload_chars(half["track_b_packing"], half["track_b_context"]) == cap

    client.calls.clear()
    tail = "This is the final turn."
    source = "a" * 3500 + tail
    baseline = default["session_summary"]
    for strategy in (default["dual_track"], baseline):
        strategy.add_turn("long", 0, source, "")
        strategy.get_context("long", "question", 4000)
    assert len(client.calls) == 1
    prompt, options = client.calls[0]
    assert tail in prompt and options["max_tokens"] == 300

    main = strategies(load_experiment_config("configs/main_sweep.yaml"))
    for strategy in (main["dual_track"], main["bm25_chunk_trackb"]):
        for turn in range(6):
            strategy.add_turn("s", turn, "needle " * 35 if turn < 5 else "short answer", "")
    assert main["dual_track"]._mem.get_context("s", "needle", cap, 0) == half["track_b_context"]
    assert main["bm25_chunk_trackb"].get_context("s", "needle", cap) == TRACK_B_HEADER + exact + TRACK_B_FOOTER
    main["session_summary"].add_turn("long", 0, source, "")
    main["session_summary"].get_context("long", "question", 4000)
    assert len(client.calls) == 1 and tail in client.calls[0][0]

    memory = main["dual_track"]._mem
    for turn in range(9):
        memory.add_turn("short", turn, "tea", "", real_session_id="short")
    for budget, ratio, count in [(1000, 0.5, 5), (2000, 0.25, 5), (4000, 0.75, 5),
                                 (6000, 0.5, 5), (12000, 0.5, 9), (24000, 0.5, 9),
                                 (4000, 0, 9), (6000, 1, 0)]:
        _, stats = memory.get_context_with_attribution("short", "tea", budget, ratio)
        packing = stats["track_b_packing"]
        assert packing["selected_unit_count"] == count
        assert packing["selected_source_indices"] == list(range(count))
        capped = 0 < ratio < 1 and budget <= 6000
        assert packing["max_selected"] == (5 if capped else None)
        assert packing["count_cap_binding"] == capped
