import json
from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts import aggregate_paper_tables
from experiments import exp_budget_sweep
from experiments.exp_budget_sweep import run_budget_sweep, select_strategies
from experiments.exp_orthogonality import check_coverage, stored_track_contexts
from experiments.exp_rebuttal import rescore
from scripts.aggregate_paper_controls import main_judge_audit_grid, matched_subset_decisions, route_task_matrix, utilization_table
from experiments.exp_masking_n500 import run_one_sample
from scripts.paired_crossover_stats import sample_map
from src.memory.track_a import EpisodicTrack
from src.memory.track_b_rag import SemanticRAGTrack
from experiments.exp_paper_controls import judge_prompt, reuse_main
from src.evaluation.longmemeval_runner import EvalSample, JUDGE_PROMPT, LongMemEvalRunner
from src.memory.baselines.strategies import FullContextStrategy
from src.models.llm_client import LLMResponse
from scripts.judge_validation_matched_control import load_stored_outputs
from src.evaluation.locomo_runner import CATEGORY_TO_TASK_TYPE, iter_locomo_turns
from src.memory.baselines.strategies import (
    BM25ChunkStrategy, BM25SessionStrategy, EmbeddingChunkStrategy,
    EmbeddingSessionStrategy, MERGED_SESSION_ID, RecentSessionsStrategy, SessionSummaryStrategy,
)


class ReleaseProtocolTests(unittest.TestCase):
    def test_selected_chunks_keep_source_order_within_each_turn(self):
        track = SemanticRAGTrack()
        track.add("s", [{"role": "user", "content": "earlier"},
                        {"role": "assistant", "content": "needle"}], turn=2)
        hits = track.search("s", "needle")
        self.assertEqual(hits[0].role, "assistant")
        context = track.format_context_block(hits, 1000)
        self.assertLess(context.index("earlier"), context.index("needle"))

    def test_small_chunk_baselines_keep_source_order_after_selection(self):
        for strategy in (BM25ChunkStrategy(chunk_size=2, chunk_overlap=0, recency_bias=0),
                         EmbeddingChunkStrategy(chunk_size=2, chunk_overlap=0, recency_bias=0)):
            if isinstance(strategy, EmbeddingChunkStrategy):
                strategy._model = Mock()
                strategy._model.encode.side_effect = lambda text, **kw: np.array([float("needle" in text)])
            strategy.add_turn("s", 0, "early words needle later", "needle reply")
            context = strategy.get_context("s", query="needle", budget_chars=500)
            self.assertEqual([line.split("] ")[-1] for line in context.splitlines()[1:-1]],
                             ["early words", "needle later", "needle reply"])

    def test_word_chunk_baselines_have_identical_budget_boundaries(self):
        dense = EmbeddingChunkStrategy()
        dense._model = Mock()
        dense._model.encode.return_value = np.array([1.0])
        bm25 = BM25ChunkStrategy()
        for strategy in (bm25, dense):
            strategy.add_turn("s", 0, "x" * 925, "")
        exact = len(bm25.get_context("s", "q", 2000))
        for budget in (exact - 1, exact, 1000):
            wanted = bm25.get_context("s", "q", budget)
            self.assertEqual(bool(wanted), budget >= exact)
            self.assertEqual(dense.get_context("s", "q", budget), wanted)

    def test_paper_strategy_names_select_existing_implementations(self):
        strategies = [SimpleNamespace(name=n) for n in ("bm25_chunk", "bm25_chunk_trackb", "dual_track")]
        for names in (["bm25_chunk_64w_top8", "bm25_trackb_full"], ["bm25_chunk", "bm25_chunk_trackb"]):
            self.assertEqual(select_strategies(strategies, names), strategies[:2])
        with self.assertRaisesRegex(ValueError, "Unknown strategies"):
            select_strategies(strategies, ["bm25_chunk", "missing_method"])

    def test_main_cli_uses_benchmark_defaults_and_accepts_explicit_overrides(self):
        class SweepStarted(Exception):
            pass

        scenarios = [
            ([], [1000, 2000, 4000, 6000], 500, 42, True),
            (["--benchmark", "locomo"], [500, 1000, 2000, 4000], 200, 42, True),
            (["--strategies", "bm25_trackb_full", "--budgets", "850", "--max_samples", "1",
              "--seeds", "3", "--no-shuffle"], [850], 1, 3, False),
        ]
        names = ("recent_sessions", "bm25_session", "session_summary", "embedding_chunk",
                 "bm25_chunk", "bm25_chunk_trackb", "dual_track")
        strategies = [SimpleNamespace(name=n) for n in names]
        for flags, budgets, size, seed, shuffle in scenarios:
            with patch("sys.argv", ["sweep", *flags]), patch.object(exp_budget_sweep, "LLMClient"), \
                    patch.object(exp_budget_sweep, "build_strategies_from_config", return_value=strategies), \
                    patch.object(exp_budget_sweep, "run_budget_sweep", side_effect=SweepStarted) as run, \
                    redirect_stdout(StringIO()), self.assertRaises(SweepStarted):
                exp_budget_sweep.main()
            cfg = run.call_args.kwargs
            self.assertEqual((cfg["budgets"], cfg["max_samples"], cfg["seed"], cfg["shuffle"]),
                             (budgets, size, seed, shuffle))
            selected = {s.name for s in cfg["strategies"]}
            if "--strategies" in flags:
                self.assertEqual(selected, {"bm25_chunk_trackb"})
            elif cfg["benchmark"] == "locomo":
                self.assertEqual(selected, {"bm25_chunk_trackb", "dual_track"})
                self.assertEqual(cfg["data_path"], "data/locomo/locomo_test.json")
            else:
                self.assertEqual(selected, set(names))

    def test_count_cap_binding_agrees_with_uncapped_context(self):
        track = SemanticRAGTrack()
        for turn, text in enumerate(["short"] * 5 + ["long " * 100, "later short"]):
            track.add("s", [dict(role="user", content=text)], turn)
        hits = track.search("s", "absent")
        for cap in (0, 130, 160, 500, 1200):
            uncapped = track.format_context_block(hits, cap)
            capped = track.format_context_block(hits, cap, max_selected=5)
            self.assertEqual(track.last_packing_stats["count_cap_binding"], capped != uncapped)
        self.assertTrue(track.last_packing_stats["count_cap_binding"])

    def test_summary_endpoint_reuses_track_a_selection_and_serialization(self):
        model = SimpleNamespace(model="fixture")
        model.chat = lambda messages, **kw: LLMResponse(
            "" if "empty session" in messages[0]["content"] else "Saved fact.", 5, 2)
        with tempfile.TemporaryDirectory() as cache:
            a = EpisodicTrack(llm_client=model, summary_cache_dir=cache)
            endpoint = SessionSummaryStrategy(llm_client=model, summary_cache_dir=cache)
            for turn, text in enumerate(("empty session", "blue ticket", "green ticket")):
                for view in (a, endpoint):
                    view.add_turn(str(turn), turn, text, "")
            for budget in (0, 1, 64, 1000):
                source = a.get_context(MERGED_SESSION_ID, "ticket", budget)
                self.assertEqual(endpoint.get_context(MERGED_SESSION_ID, "ticket", budget), source)
                self.assertEqual(a.last_packing_stats, endpoint.last_packing_stats)
                self.assertNotIn("[Session 1]", source)
            self.assertIn("[Session 2]", source)

    def test_valid_mixed_labels_and_parser_fallbacks(self):
        rows = [dict(task_type="temporal", label=label, parse_success=parsed)
                for label, parsed in [("STATE", True), ("MIXED", True), ("MIXED", False)]]
        counts = route_task_matrix(rows)["temporal"]
        self.assertEqual((counts["MIXED"], counts["total"]), (2, 3))
        self.assertEqual((counts["valid_mixed"], counts["parse_fallbacks"]), (1, 1))

    def test_offline_summaries_precede_all_answers(self):
        runner = LongMemEvalRunner("")
        runner._data = [EvalSample(sid, "Question?", "answer", "temporal",
                                   [dict(role="user", content=sid, session_id=sid)], [sid])
                        for sid in ("first", "second")]
        events = []
        def chat(messages, **kwargs):
            summary = kwargs["max_tokens"] == 300
            events.append("summary" if summary else "answer")
            return LLMResponse("fact" if summary else "answer", 2, 1)
        model = SimpleNamespace(model="fixture", chat=chat)
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                strategy = SessionSummaryStrategy(llm_client=model, summary_cache_dir=directory)
                runner.evaluate(strategy, model, budget_chars=1000, judge_mode="string")
        self.assertEqual(events, ["summary", "summary"] + ["answer"] * 4)

    def test_table_exports_include_reused_comparison_rows(self):
        def save(root, name, obj):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj))
        def sweep(correct):
            return dict(budgets=[4000], sample_correctness=[[
                dict(sample_id="q", is_correct=correct, input_tokens=40)]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "tables"
            save(root, "main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json",
                 {"dual_track": sweep(True)})
            save(root, "trackb_deepseek/trackb_fullbudget_results.json",
                 {"bm25_chunk_trackb": sweep(False)})
            for policy in ("dense_split", "dense_trackb"):
                save(root, f"dense_matched/raw/{policy}_b4000.jsonl",
                     dict(sample_id="q", is_correct=True))
            save(root, "paper_controls/tables.json", {"table_length_matched_concentration":
                 {"4000": dict(n=1, accuracy_pct=0)}})
            args = ["aggregate", "--allow-missing", "--results-dir", str(root), "--output-dir", str(out)]
            with patch("sys.argv", args), redirect_stdout(StringIO()):
                aggregate_paper_tables.main()
            dense = json.loads((out / "table_03_dense_matched.json").read_text())
            self.assertEqual(dense["bm25_delta_pp_b4000"], 100)
            self.assertEqual(dense["dense_delta_pp_b4000"], 0)
            length = json.loads((out / "table_12_length_matched.json").read_text())["4000"]
            self.assertEqual((length["split_accuracy_pct"], length["full_cap_accuracy_pct"],
                              length["accuracy_pct"], length["split_minus_length_matched_pp"]), (100, 0, 0, 100))

    def test_grid_reuses_verified_subset_verdicts(self):
        record = dict(sample_id="q", prediction="paper answer", gold_answer="gold",
                      question="Question?", task_type="temporal", is_correct=True)
        cells = {(name, 4000): [dict(record)] for name in ("ratio50", "trackb1024")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "_partial").mkdir()
            for strategy in ("dual_track", "bm25_chunk_trackb"):
                audit = dict(strategy=strategy, budget=4000, judging=dict(judge_model="gpt-4o-mini"),
                             samples=[dict(record, main_judge_correct=True, is_correct=False)])
                path = root / "_partial" / f"matched_judge_{strategy}_b4000_gpt-4o-mini.json"
                path.write_text(json.dumps(audit))
            labels = matched_subset_decisions(root, cells, {"q"})
            grid = main_judge_audit_grid(cells, {"q"}, root, labels)
            for condition in ("ratio50", "trackb1024"):
                self.assertEqual(grid[condition]["independent_judge_audit_subset"]["correct"], 0)
            cells["ratio50", 4000][0]["prediction"] = "different answer"
            with self.assertRaisesRegex(ValueError, "Matched audit differs"):
                matched_subset_decisions(root, cells, {"q"})

    def test_appendix_packing_observes_the_split_limit(self):
        from experiments.rebuttal_memory import pack
        views = {"summary": [], "chunk": [(i, f"unit-{i}") for i in range(8)]}
        for char_mode, budget, split_count in ((True, 6000, 5), (True, 12000, 8), (False, 1024, 5)):
            for summary_pct, count in ((0, 8), (50, split_count)):
                shares = [("summary", summary_pct), ("chunk", 100 - summary_pct)]
                context, _ = pack(views, shares, budget, len, character_budget=char_mode)
                self.assertEqual(context.count("unit-"), count)

    def test_history_occurrences_and_summary_cache(self):
        raw = dict(haystack_session_ids=["a", "b", "a"],
                   haystack_dates=["2020/01/03 01:00", "2020/01/02 01:00", "2020/01/01 01:00"],
                   haystack_sessions=[[{"role": "user", "content": "last"}],
                                      [{"role": "assistant", "content": "middle"}],
                                      [{"role": "user", "content": "first"}]])
        history, ids = LongMemEvalRunner._normalize_history(raw)
        pairs = LongMemEvalRunner._iter_turn_pairs(history)
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual([(p["session_id"], p["user"], p["assistant"]) for p in pairs],
                         [(ids[0], "first", ""), (ids[1], "", "middle"), (ids[2], "last", "")])
        track = EpisodicTrack()
        baseline = SessionSummaryStrategy(use_llm_summary=False)
        for sid, offset in [("one", 2), ("two", 500)]:
            for strategy in (track, baseline):
                strategy.add_turn(sid, offset, "Tea?", "Yes.")
                strategy.add_turn(sid, offset + 1, "", "At noon.")
        left = track._build_session_text(track._states["one"])
        right = baseline._format_session_for_summary("two")
        self.assertEqual(left, right)
        self.assertEqual(track._cache_key(left), track._cache_key(right))
        self.assertNotIn("User: "+chr(10), left)

    def test_chunk_tail_and_utilization(self):
        track = SemanticRAGTrack()
        source = "x" * 1100
        self.assertEqual(list(map(len, track._chunk_text(source))), [1024, 204])
        split = [dict(sample_id=i, memory_context_len=20, attribution={
            "track_a_packing": dict(serialized_chars=20, selected_unit_count=1),
            "track_b_packing": dict(budgeted_chars=0, selected_unit_count=0),
        }) for i in (0, 1)]
        chunks = [dict(sample_id=i, memory_context_len=chars + 37, packing_stats=dict(
            budgeted_chars=chars, selected_unit_count=count)) for i, (chars, count) in enumerate([(100, 1), (200, 3)])]
        cells = {("ratio50", 4000): split, ("trackb1024", 4000): chunks}
        stats = utilization_table(cells)["4000"]
        self.assertEqual((stats["concentration_chunks_per_item"], stats["concentration_injected_chars"]), (2, 150))
        self.assertEqual(stats["track_a_fill"], 20 / 2000)
        for row, binding in zip(split, (True, False)):
            row["attribution"]["track_b_packing"]["count_cap_binding"] = binding
        self.assertEqual(utilization_table(cells)["4000"]["split_count_cap_binding_items"], 1)
        split[0]["attribution"]["use_router"] = True
        with self.assertRaisesRegex(ValueError, "static 50:50"):
            utilization_table(cells)
        split[0]["attribution"]["use_router"] = False
        del split[0]["attribution"]["track_a_packing"]["selected_unit_count"]
        with self.assertRaisesRegex(ValueError, "Missing Table 6 packing counts"):
            utilization_table(cells)

    def test_coverage_uses_stored_context_and_whole_keywords(self):
        self.assertEqual(check_coverage("cat", "a category", "A cat!")["source"], "B_only")
        for answer in ["I don't know", "I don’t know.", "I do not know"]:
            self.assertEqual(check_coverage(answer, "", "")["source"], "unknown")
        self.assertEqual(check_coverage("cat dog", "cat", "")["source"], "A_only")
        with self.assertRaisesRegex(ValueError, "Missing stored track contexts"):
            stored_track_contexts(dict(sample_id="q"))
        self.assertEqual(stored_track_contexts(dict(sample_id="q", attribution=dict(
            track_a_context="", track_b_context=""))), ("", ""))

    def test_masking_reuses_recorded_parts_and_full_answer(self):
        sample = SimpleNamespace(sample_id="q", task_type="temporal", question="When?", gold_answer="Monday")
        source = dict(sample_id="q", task_type="temporal", condition="ratio50", question="When?",
                      gold_answer="Monday", nominal_budget_chars=4000, prediction="Monday", is_correct=True,
                      memory_context="summaries\n\nchunks",
                      attribution=dict(track_a_context="summaries", track_b_context="chunks"))
        model = Mock(model="fixture")
        model.chat.return_value = LLMResponse("Monday", 1, 1)
        runner = LongMemEvalRunner("")
        row = run_one_sample(None, runner, model, sample, 4000, "dual_track", source=source)
        self.assertEqual(row["answer"], "Monday")
        model.chat.assert_not_called()
        for name, retained in [("dual_track_mask_a", "chunks"), ("dual_track_mask_b", "summaries")]:
            row = run_one_sample(None, runner, model, sample, 4000, name, source=source)
            self.assertEqual(row["mem_ctx_preview"], retained)

    def test_paired_statistics_reject_duplicate_ids(self):
        row = {"sample_id": "q", "is_correct": True}
        cell = {"budgets": [4000], "sample_correctness": [[row, row]]}
        with self.assertRaisesRegex(ValueError, "Duplicate ID"):
            sample_map({"dual_track": cell}, "dual_track", 4000)

    def test_controls_reuse_main_answers_and_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main, control, data = (root / name for name in ("main.json", "track.json", "questions.json"))
            data.write_text(json.dumps([dict(question_id="q", question="Where?", answer="Paris", history=[])]))
            def cell(answer):
                return dict(budgets=[4000], sample_correctness=[[dict(
                    sample_id="q", prediction=answer, memory_context="original context",
                    attribution={"track_a_context": "original summary"}, is_correct=True,
                )]])
            main.write_text(json.dumps({"raw_runs": [{"dual_track": cell("split"), "session_summary": cell("summary")}]}))
            control.write_text(json.dumps({"bm25_chunk_trackb": cell("chunks")}))
            reuse_main(SimpleNamespace(main=main, trackb=control, data_path=data, output_dir=root, judge_model="judge"))
            for name, answer in [("ratio50", "split"), ("tracka1024", "summary"), ("trackb1024", "chunks")]:
                row = json.loads((root / "raw" / f"{name}_b4000.jsonl").read_text())
                self.assertEqual((row["question"], row["prediction"], row["memory_context"]),
                                 ("Where?", answer, "original context"))

    def test_full_history_prompt_preserves_every_message(self):
        runner = LongMemEvalRunner("")
        history = [dict(session_id=sid, role=role, content=text) for sid, role, text in [
            ("first", "assistant", "opening"), ("first", "user", "question one"),
            ("second", "user", "question two"), ("second", "assistant", "closing"),
        ]]
        runner._data = [EvalSample(sample_id="q", question="What happened?", gold_answer="yes",
                                   history=history, session_ids=["first", "second"], task_type="temporal")]
        model = Mock()
        model.chat.return_value = LLMResponse("yes", 20, 1)
        result = runner.evaluate(FullContextStrategy(), model, budget_chars=1,
                                 judge_mode="string", store_context=True).samples[0]
        self.assertEqual(result.memory_context.splitlines(), [
            "[Full Conversation]", "[Session 1]", "Assistant: opening", "User: question one",
            "[Session 2]", "User: question two", "Assistant: closing", "[/Full Conversation]",
        ])

    def test_question_inclusive_changes_only_question_visibility(self):
        row = dict(task_type="temporal", question="When exactly?", gold_answer="Tuesday", prediction="Tuesday")
        original = judge_prompt(row, "questionless")
        self.assertEqual(original, JUDGE_PROMPT.format(gold="Tuesday", prediction="Tuesday"))
        self.assertEqual(judge_prompt(row, "question_inclusive").replace("Question: When exactly?\n", ""), original)
        row["task_type"] = "abstention"
        self.assertIn(row["question"], judge_prompt(row, "questionless"))
        self.assertEqual(judge_prompt(row, "questionless"), judge_prompt(row, "question_inclusive"))

    def test_locomo_preserves_sides_and_session_boundaries(self):
        history = [
            ("s1", "assistant", "first"), ("s1", "user", "second"),
            ("s1", "user", "third"), ("s1", "assistant", "fourth"),
            ("s1", "user", "fifth"), ("s2", "assistant", "sixth"),
            ("s2", "assistant", "seventh"), ("s2", "user", "last"),
        ]
        messages = [dict(session_id=sid, role=role, content=text) for sid, role, text in history]
        turns = [(t["session_id"], t["user"], t["assistant"]) for t in iter_locomo_turns(messages)]
        self.assertEqual(turns, [
            ("s1", "", "first"), ("s1", "second", ""), ("s1", "third", "fourth"),
            ("s1", "fifth", ""), ("s2", "", "sixth"), ("s2", "", "seventh"), ("s2", "last", ""),
        ])
        self.assertEqual([CATEGORY_TO_TASK_TYPE[i] for i in (1, 3, 4)], ["multi_hop", "open_domain", "single_hop"])

    def test_baseline_context_is_independent_of_source_ids(self):
        def render(sid):
            encoder = Mock()
            encoder.encode.return_value = np.array([1.0])
            dense_session, dense_chunk = EmbeddingSessionStrategy(), EmbeddingChunkStrategy()
            dense_session._model = dense_chunk._model = encoder
            sparse = BM25SessionStrategy()
            strategies = [RecentSessionsStrategy(), sparse, BM25ChunkStrategy(), dense_chunk,
                          dense_session, SessionSummaryStrategy(use_llm_summary=False)]
            contexts = {}
            for strategy in strategies:
                strategy.add_turn("background", 0, "Tea was served.", "Yes.")
                strategy.add_turn(sid, 1, "The meeting was on Tuesday.", "Thank you.")
                context = strategy.get_context(MERGED_SESSION_ID, "When was the meeting?", 4000)
                self.assertIn("Tuesday", context)
                self.assertNotIn(sid, context)
                contexts[strategy.name] = context
            self.assertNotIn(sid, sparse._session_terms[sid])
            return contexts

        self.assertEqual(render("answer_hidden"), render("unrelated_source"))

    def test_frozen_audit_alignment_and_abstention_prompt(self):
        row = dict(sample_id="q", task_type="abstention", gold_answer="Unknown", prediction="I don't know",
                   is_correct=True)
        cell = dict(budgets=[4000], sample_correctness=[[row]])
        with tempfile.TemporaryDirectory() as directory:
            main, control = (Path(directory) / name for name in ("main.json", "control.json"))
            main.write_text(json.dumps({"raw_runs": [{"dual_track": cell}]}))
            def load(rows):
                control.write_text(json.dumps({"bm25_chunk_trackb": dict(cell, sample_correctness=[rows])}))
                return load_stored_outputs(4000, ["q"], main, control, {"q": "Where is the otter?"})

            saved = load([row])["dual_track"][0]
            judge = Mock(model="independent")
            judge.chat.return_value.content = "YES"
            self.assertEqual(rescore(saved, judge)["prediction"], row["prediction"])
            self.assertIn(saved["question"], judge.chat.call_args.kwargs["messages"][0]["content"])
            rescore(dict(saved, task_type="temporal"), judge)
            self.assertNotIn(saved["question"], judge.chat.call_args.kwargs["messages"][0]["content"])
            with self.assertRaisesRegex(ValueError, "Duplicate question"):
                load([row, row])
            with self.assertRaisesRegex(ValueError, "missing"):
                load([])
            with self.assertRaisesRegex(ValueError, "Unknown question"):
                load([row, dict(row, sample_id="extra")])

    def test_sweep_reuses_only_compatible_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data, cfg = base / "data.json", base / "settings.yaml"
            data.write_text("[]")
            cfg.write_text("strategies: {}")
            options = dict(strategies=[SimpleNamespace(name="dual_track")], answer_llm=None, judge_llm=None,
                           data_path=str(data), config_path=str(cfg), budgets=[1000], max_samples=1,
                           output_dir=str(base), sample_ids=["q"])

            def worker(command, **kwargs):
                params = json.loads(Path(command[-1]).read_text())
                result = dict(accuracy=1, avg_input_tokens=3, task_breakdown={}, sample_correctness=[])
                Path(params["result_path"]).write_text(json.dumps(result))
                return SimpleNamespace(returncode=0)

            with redirect_stdout(StringIO()), patch("experiments.exp_budget_sweep.subprocess.run", side_effect=worker) as execute:
                first = run_budget_sweep(**options)
                self.assertEqual(run_budget_sweep(**options), first)
                for change in [dict(sample_ids=["other"]), dict(max_samples=2), dict(shuffle=True),
                               dict(judge_mode="exact"), dict(judge_model_name="other")]:
                    with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Incompatible checkpoint"):
                        run_budget_sweep(**dict(options, **change))
                cfg.write_text("strategies: {dual_track: {top_k: 5}}")
                with self.assertRaisesRegex(ValueError, "Incompatible checkpoint"):
                    run_budget_sweep(**options)
                cfg.write_text("strategies: {}")
                data.write_text("[{}]")
                with self.assertRaisesRegex(ValueError, "Incompatible checkpoint"):
                    run_budget_sweep(**options)
                data.write_text("[]")
                checkpoint = next((base / "_partial").glob("partial_*.json"))
                legacy = json.loads(checkpoint.read_text())
                del legacy["run_settings"]
                checkpoint.write_text(json.dumps(legacy))
                with self.assertRaisesRegex(ValueError, "Incompatible checkpoint"):
                    run_budget_sweep(**options)
                execute.assert_called_once()
