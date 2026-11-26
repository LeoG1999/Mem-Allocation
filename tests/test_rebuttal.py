"""End-to-end offline checks; the fake transport never sends a request."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import exp_rebuttal as experiment, rebuttal_analysis as analysis
from experiments import exp_paper_controls as controls
from experiments.rebuttal_memory import pack, thin_history
from scripts.aggregate_paper_controls import index_raw, main_judge_audit_grid
from src.memory.track_b_rag import TRACK_B_HEADER, TRACK_B_FOOTER
from src.models.llm_client import LLMResponse
from tokenizers import Tokenizer, models, pre_tokenizers


class LocalClient:
    def __init__(self, model):
        self.model = model
        self.api_model = model

    def chat(self, messages, **kwargs):
        text = messages[0]["content"]
        if text.startswith("Extract durable"):
            reply = '[["Alex","lives in","Paris"]]'
        elif "EVIDENCE:" in text:
            reply = "EVIDENCE"
        elif text.startswith("You will receive a conversation"):
            reply = "Alex lives in Paris."
        elif "YES" in text:
            reply = "YES"
        else:
            reply = "Paris"
        return LLMResponse(reply, 8, 2)


class RebuttalChecks(unittest.TestCase):
    def test_cli_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = []
            for i in range(6):
                sid = f"s{i % 5}"
                raw.append(
                    dict(
                        question_id=f"q{i}",
                        question_type="single-session-user",
                        question=f"Where does person {i} live?",
                        answer="Paris",
                        question_date="2020/02/02",
                        answer_session_ids=[sid],
                        haystack_session_ids=[sid, "background"],
                        haystack_dates=["2020/01/02 10:00", "2020/01/01 10:00"],
                        haystack_sessions=[
                            [
                                dict(role="user", content="I live in Paris.", has_answer=True),
                                dict(role="assistant", content="Okay."),
                            ],
                            [
                                dict(role="user", content="Flights are delayed."),
                                dict(role="assistant", content="Okay."),
                            ],
                        ],
                    )
                )
            data = root / "long.json"
            data.write_text(json.dumps(raw))
            self.assertNotIn("has_answer", json.dumps(thin_history(raw[0], 0, 42)))
            self.assertEqual(thin_history(raw[0], 0, 42)["haystack_session_ids"], ["s0"])
            full = thin_history(raw[0], None, 42)
            self.assertEqual(full["question"], raw[0]["question"])
            self.assertEqual(full["question_date"], raw[0]["question_date"])
            expected, ids = experiment.LongMemEvalRunner._normalize_history(raw[0])
            actual, actual_ids = experiment.LongMemEvalRunner._normalize_history(full)
            self.assertEqual(actual_ids, ids)
            self.assertEqual(actual, [{k: msg[k] for k in ("role", "content", "session_id")}
                                      for msg in expected])
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            token_file = root / "tokenizer.json"
            tokenizer.save(str(token_file))
            for name in experiment.BUDGETS:
                ids_arg = []
                file = data
                if name == "locomo":
                    file = root / "locomo.json"
                    file.write_text(
                        json.dumps(
                            [
                                {
                                    "sample_id": "c",
                                    "conversation": {
                                        "speaker_a": "A",
                                        "session_1": [
                                            {"speaker": "A", "text": "I live in Paris."},
                                            {"speaker": "B", "text": "Noted."},
                                        ],
                                    },
                                    "qa": [{"question": "Where?", "answer": "Paris"}],
                                }
                            ]
                        )
                    )
                    ids = root / "ids.json"
                    ids.write_text('["c__q0"]')
                    ids_arg = ["--ids", str(ids)]
                out = root / name
                argv = [
                    "generate",
                    name,
                    "--data",
                    str(file),
                    "--output",
                    str(out),
                    "--cache",
                    str(root / "cache"),
                    "--tokenizer",
                    str(token_file),
                    "--background",
                    "0",
                    "full",
                ] + ids_arg
                with (
                    patch.object(sys, "argv", ["fixture"] + argv),
                    patch.object(experiment, "LLMClient", LocalClient),
                    patch.object(experiment, "route", wraps=experiment.route) as classify,
                ):
                    experiment.main()
                if name == "token":
                    self.assertEqual(classify.call_count, len(raw))
                answers = out / "answers.jsonl"
                rows = [json.loads(line) for line in answers.read_text().splitlines()]
                self.assertTrue(rows and (out / "complete.json").is_file())
                self.assertTrue(all(r["budget_used"] <= r["applied_cap"] for r in rows))
                if name == "joint":
                    for left, right in zip(rows[::2], rows[1::2]):
                        self.assertEqual(left["budget_used"], right["applied_cap"])
                if name == "history":
                    self.assertEqual({r["history_level"] for r in rows}, {"0", "full"})
                if name == "views":
                    self.assertTrue(all(r["context_chars"] > 0 for r in rows))
                rescored = out / "scored.jsonl"
                argv = ["rescore", "--answers", str(answers), "--output", str(rescored)]
                with (
                    patch.object(sys, "argv", ["fixture"] + argv),
                    patch.object(experiment, "LLMClient", LocalClient),
                ):
                    experiment.main()

                def report(command):
                    buf = io.StringIO()
                    with (
                        patch.object(
                            sys, "argv", ["fixture", command, str(rescored), "--judge", "independent_correct"]
                        ),
                        contextlib.redirect_stdout(buf),
                    ):
                        analysis.main()
                    return json.loads(buf.getvalue())

                summary = report("summarize")
                self.assertTrue(summary)
                if name == "token":
                    self.assertEqual(summary[0]["router_cost_change_pct"], 100)
                    timed = next(r for r in summary if r["budget"] == 1024)
                    self.assertEqual(timed["policies"]["router"]["latency_modes"], ["end_to_end"])
                if name == "views":
                    comparisons = summary[0]["mixture_minus_component_pp"]
                    self.assertEqual(set(comparisons["extract+chunks"]), {"extract", "chunks"})
                    self.assertEqual(set(comparisons["summary+chunks+facts"]), {"summary", "chunks", "facts"})
                if name == "audit":
                    decisions = report("replay")["out_of_fold"]
                    self.assertEqual(len({r["sample_id"] for r in decisions}), 6)
                    folds = {r["sample_id"]: r["fold"] for r in decisions}
                    self.assertEqual(folds["q0"], folds["q5"])
                    # Exercise the logistic path as well as constant-label folds.
                    train_y = {a: [True, False, True] for a in analysis.ACTIONS}
                    pred = analysis.fit_scores(
                        ["where is it", "which fruit", "where now"], train_y, ["where"]
                    )
                    self.assertTrue(all(0 < p[0] < 1 for p in pred.values()))

    def test_character_and_token_caps_use_distinct_wrapper_accounting(self):
        views = {"chunk": [(0, "a" * 60)]}
        shares = [("chunk", 100)]
        block, used = pack(views, shares, 60, len, character_budget=True)
        self.assertEqual(block, TRACK_B_HEADER + "a" * 60 + TRACK_B_FOOTER)
        self.assertEqual(used, 60)
        self.assertEqual(pack(views, shares, 60, len), ("", 0))

    def test_frozen_router_generates_and_audits_its_own_answer(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(controls, "LLMClient", LocalClient):
            root = Path(directory)
            data = root / "data.json"
            data.write_text(json.dumps([dict(
                question_id="q", question_type="single-session-user", question="Where?",
                answer="Paris", haystack_session_ids=["a"], haystack_dates=["2020/01/01 10:00"],
                haystack_sessions=[[dict(role="user", content="Paris"), dict(role="assistant", content="OK")]],
            )]))
            routes = root / "routes/route_b.jsonl"
            def run(argv):
                args = controls.build_parser().parse_args(argv)
                with contextlib.redirect_stdout(io.StringIO()):
                    args.func(args)
            run(["route", "--data-path", str(data), "--output", str(routes)])
            run(["generate", "--conditions", "router", "--budgets", "4000", "--data-path", str(data),
                 "--output-dir", str(root), "--routes", str(routes), "--summary-cache-dir", str(root / "cache")])
            output = root / "raw/router_b4000.jsonl"
            row = json.loads(output.read_text())
            self.assertEqual(row["condition_spec"]["track_a_ratio"], 0.25)
            self.assertEqual(row["attribution"]["track_a_cap_chars"], 1000)
            run(["rejudge", "--inputs", str(output), "--prompt-mode", "questionless",
                 "--judge-model", "gpt-4o-mini", "--output-dir", str(root / "rejudge")])
            grid = main_judge_audit_grid(index_raw(root), {"q"}, root)
            self.assertEqual(grid["router"]["independent_judge_audit_subset"]["n"], 1)

            run(["generate", "--conditions", "trackb1024", "tracka1024", "ratio25", "ratio33",
                 "ratio50", "ratio67", "ratio75", "--budgets", "4000", "--data-path", str(data),
                 "--output-dir", str(root), "--summary-cache-dir", str(root / "cache")])
            frozen = root / "frozen.jsonl"
            experiment.prepare_audit(data, root, frozen)
            rows = [json.loads(line) for line in frozen.read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            direct = next(r for r in rows if r["condition"] == "router")
            self.assertEqual((direct["prediction"], direct["context_sha256"], direct["selected_ratio"]),
                             (row["prediction"], row["context_sha256"], 25))
            output.write_text("")
            with self.assertRaisesRegex(ValueError, "Incomplete frozen"):
                experiment.prepare_audit(data, root, root / "incomplete.jsonl")
            self.assertFalse((root / "incomplete.jsonl").exists())

    def test_shared_sources_cannot_cross_folds(self):
        rows = [dict(question_id=q, haystack_session_ids=[q], answer_session_ids=support)
                for q, support in [("a", ["s"]), ("b", ["s", "t"]), ("c", ["t"]), ("d", [])]]
        groups = experiment.source_history_groups(rows)
        self.assertEqual(groups["a"], groups["c"])
        self.assertNotEqual(groups["a"], groups["d"])

    def test_repeated_session_ids(self):
        item = dict(question_id="q", question_type="factual", question="When?", answer="Today",
                    question_date="2020/01/04", answer_session_ids=["s"],
                    haystack_session_ids=["s", "b", "b"], haystack_sessions=[[], [], []],
                    haystack_dates=["2020/01/01 12:00", "2020/01/03 12:00", "2020/01/02 12:00"])
        full = thin_history(item, None, 42)
        short = thin_history(item, 1, 42)
        self.assertEqual(full["haystack_session_ids"], ["0:s", "2:b", "1:b"])
        self.assertEqual(len(short["haystack_session_ids"]), 2)
        self.assertIn("0:s", short["haystack_session_ids"])

    def test_alignment(self):
        with self.assertRaises(ValueError):
            analysis.paired({"a": True}, {"b": True})
        family = [{"p_exact": p} for p in [0.02, 0.01, 0.8, 0.8]]
        analysis.holm(family)
        self.assertEqual([round(r["p_holm"], 2) for r in family], [0.06, 0.04, 1, 1])
