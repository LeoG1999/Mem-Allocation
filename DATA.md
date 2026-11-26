# Benchmark data

The experiments use LongMemEval-S for the main comparison and LoCoMo for cross-benchmark evaluation. Download the benchmarks from the authors and place the JSON files at the paths below. The default LoCoMo transfer samples 200 questions from the complete dataset with seed 42.

| Benchmark | Local path | Source |
|---|---|---|
| LongMemEval-S | `data/longmemeval/longmemeval_s.json` | [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval) |
| LoCoMo | `data/locomo/locomo_test.json` | [LoCoMo repository](https://github.com/snap-research/locomo) |

## Verify the inputs

From the repository root, run:

```bash
python scripts/check_data.py
```

The checker compares each local file with its expected SHA-256 digest.

| Dataset | SHA-256 |
|---|---|
| LongMemEval-S | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| LoCoMo | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |

The source repositories provide the dataset terms and redistribution guidance. For model and budget settings, see the [experiment configuration](configs/experiment_config.yaml).
