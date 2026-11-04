# SPARK benchmarks

## Install

Use Python 3.10 and separate environments for quality and MPC execution:

```bash
python3.10 -m venv .venv-quality
. .venv-quality/bin/activate
pip install -r requirements-transformers.txt
python compensation.py --output outputs/compensation.json
```

```bash
python3.10 -m venv .venv-mpc
. .venv-mpc/bin/activate
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements-crypten.txt
python -m effciency.block_benchmark --shape small --tokens 4 --output outputs/block.json
python -m effciency.operator_benchmark --tokens 4 --width 32 --repeats 1 --output outputs/operators.json
python -m effciency.decoder_benchmark --device cpu --smoke --repeats 1 --output outputs/decoder.json
```

## Observation and compensation communication

```bash
python -m effciency.controller_benchmark --tokens 4 8 16 --repeats 3 --output outputs/controller
python -m effciency.controller_benchmark --tokens 16 --methods selected_only \
  --serial-ranking --output outputs/controller_serial
python -m effciency.ranking_benchmark --tokens 16 32 --output outputs/ranking
python -m effciency.compensation_benchmark --controller-rows outputs/controller/rank0.json \
  --output outputs/scale
```

## Quality benchmarks

Supply a local Llama-3.1-8B-Instruct or Mistral-7B-Instruct-v0.3 checkpoint and its
matching GELU adapter using `--adapter /path/to/adapter`, or use a merged GELU checkpoint.
Weights and datasets are supplied separately. Run from this directory:

```bash
python -m performance.harness --model /path/to/model --adapter /path/to/adapter \
  --recipe spark --output outputs/static
python -m performance.harness --model /path/to/model --adapter /path/to/adapter \
  --mode paired --policy performance/configs/llama.json --output outputs/paired
python -m performance.longbench --model /path/to/model --adapter /path/to/adapter \
  --data /path/to/multi_news.jsonl --output outputs/multi_news
python -m performance.longbench --model /path/to/model --adapter /path/to/adapter \
  --data /path/to/multi_news.jsonl --policy performance/configs/llama.json \
  --output outputs/multi_news_paired
```

## SST-2 observation windows

```bash
python -m performance.legacy_sst2 --model /path/to/model --adapter /path/to/adapter \
  --data /path/to/sst2/validation.parquet --windows 4 8 16 --output outputs/windows
```