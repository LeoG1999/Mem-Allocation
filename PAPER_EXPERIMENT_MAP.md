# Paper experiment map

Table numbers and final JSON filenames follow the camera-ready paper.

| Paper | Comparison | Entry point |
|---|---|---|
| Tables 1, 5, 7 · Figure 2 | Main budget sweep, input tokens and paired uncertainty | [Main comparison](scripts/reproduce/01_main.sh) |
| Tables 2, 3, 6 | Chunk size, dense retrieval and budget utilization | [Retriever and packing controls](scripts/reproduce/02_chunk_and_dense.sh) |
| Table 4 | Strategy definitions | [Memory implementations](src/memory/) |
| Table 8 | Full-set independent audit of the matched pair | [Rescore saved main-sweep answers](scripts/reproduce/04_full_audit.sh) |
| Tables 9, 10, 11, 12 | Subset audits, question-inclusive judging and realized-length matching | [Judge and length controls](scripts/reproduce/04_judges_and_length.sh) |
| Tables 13, 14, 15 · Figures 4, 6 | Masking and evidence coverage | [Masking and coverage](scripts/reproduce/05_masking_and_coverage.sh) |
| Table 16 | Extended budgets and full history | [Extended budget sweep](scripts/reproduce/06_extended_budget.sh) |
| Tables 17, 18, 19 · Figure 3 | Static ratios, oracle selection and router decisions | [Allocation and routing](scripts/reproduce/03_allocation_router.sh) |
| Table 20 · Appendix G | Held-out router selection, heuristic replay and action-matched oracle | `prepare-audit` → `rescore` → `replay` |
| Appendix G | Frozen router transfer on LoCoMo | `generate locomo` |
| Table 21 | 256-character chunks with per-question length matching and independent judging | `generate joint` → `rescore` → `summarize` |
| Appendix H.2 | Token budgets, cost and latency | `generate token` |
| Table 22 | Extractive summaries and structured facts | `generate views` |
| Appendix H.4 | History length with supporting sessions held fixed | `generate history` |
| Table 23 · Figure 5 | LoCoMo budget sweep | [LoCoMo comparison](scripts/reproduce/07_locomo.sh) |

## Running the controls

The [experiment driver](experiments/exp_rebuttal.py) generates answers and rescores saved predictions. The [analysis module](experiments/rebuttal_analysis.py) computes paired statistics and replays routing policies over the same candidate answers. For command-line options:

```bash
python -m experiments.exp_rebuttal generate --help
python -m experiments.exp_rebuttal rescore --help
python -m experiments.rebuttal_analysis --help
```

Run the main workflows with `bash scripts/reproduce/run_all.sh`. To include the held-out selector, independent router audit, Appendix H controls, and LoCoMo transfer, supply the local tokenizer and request the appendix workflows:

```bash
ANSWER_TOKENIZER=/path/to/tokenizer.json bash scripts/reproduce/run_all.sh --with-appendix
```

After completing the numbered main workflows, the appendix can also be run separately with `bash scripts/reproduce/07_appendix.sh` and aggregated with `bash scripts/reproduce/08_aggregate.sh --with-appendix`. Use a fresh appendix output directory. It independently scores the seven stored 4k candidate answers and the frozen router answer for each question. Folds keep identical source histories together and keep questions linked by shared supporting sessions in the same group. Only the splitter sees these group assignments. The selector receives question text. The `replay --groups` option accepts an explicit question-ID-to-source-group mapping.

Independent rescoring defaults to GPT-4o-mini and reads `OPENAI_API_KEY`. For token-budget and history-length runs, pass the answer model's local `tokenizer.json` with `--tokenizer`. LoCoMo uses a seeded sample of 200 questions by default and accepts an explicit list with `--ids`.

The [aggregation script](scripts/reproduce/08_aggregate.sh) builds tables and figures for the numbered shell workflows.
