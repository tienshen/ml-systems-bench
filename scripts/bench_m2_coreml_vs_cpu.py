#!/usr/bin/env python3
"""
Benchmark CPU-only vs CoreML+CPU on Apple M2 for a given ONNX model.

Example:

  # BERT base
  python scripts/bench_m2_coreml_vs_cpu.py \
    --model-name bert-base-uncased

  # tiny random BERT
  python scripts/bench_m2_coreml_vs_cpu.py \
    --model-name hf-internal-testing/tiny-random-bert

By default, it looks for an ONNX file at:
  models/<model-name-with-slashes-replaced>.onnx

You can override that with --onnx-path.
"""

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


@dataclass
class M2ExperimentConfig:
    id: str
    provider_mode: Literal["cpu_only", "coreml_plus_cpu"]
    seq_len: int
    batch_size: int
    n_warmup: int = 5
    n_iters: int = 50


def make_dummy_inputs(tokenizer, seq_len: int, batch_size: int) -> Dict[str, Any]:
    text = ["hello world"] * batch_size
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {k: v for k, v in enc.items()}


def build_session(onnx_path: Path, provider_mode: str, verbose_ort: bool) -> ort.InferenceSession:
    so = ort.SessionOptions()
    # 2 = warnings+errors only, 1 = info+warnings+errors
    so.log_severity_level = 1 if verbose_ort else 2

    if provider_mode == "cpu_only":
        providers = ["CPUExecutionProvider"]
    elif provider_mode == "coreml_plus_cpu":
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        raise ValueError(f"Unknown provider_mode: {provider_mode}")

    sess = ort.InferenceSession(onnx_path.as_posix(), sess_options=so, providers=providers)
    print(f"[INFO] Requested providers: {providers}")
    print(f"[INFO] Effective providers: {sess.get_providers()}")
    return sess


def run_ort_benchmark(
    cfg: M2ExperimentConfig,
    onnx_path: Path,
    tokenizer,
    model_name: str,
    verbose_ort: bool,
) -> Dict[str, Any]:
    print(f"\n=== Running {cfg.id} ({model_name}) ===")
    print(
        f"mode={cfg.provider_mode}, seq_len={cfg.seq_len}, "
        f"batch_size={cfg.batch_size}, iters={cfg.n_iters}"
    )

    sess = build_session(onnx_path, cfg.provider_mode, verbose_ort)
    inputs = make_dummy_inputs(tokenizer, cfg.seq_len, cfg.batch_size)

    # Warmup
    for _ in range(cfg.n_warmup):
        _ = sess.run(None, inputs)

    # Timed runs
    times = []
    for _ in range(cfg.n_iters):
        t0 = time.perf_counter()
        _ = sess.run(None, inputs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    latency_p50_ms = np.percentile(times, 50) * 1000
    latency_p95_ms = np.percentile(times, 95) * 1000
    throughput = (cfg.batch_size * cfg.n_iters) / times.sum()

    metrics = {
        "model_name": model_name,
        "latency_p50_ms": latency_p50_ms,
        "latency_p95_ms": latency_p95_ms,
        "throughput_samples_per_s": throughput,
    }
    print(
        f"[RESULT] p50={latency_p50_ms:.2f} ms, p95={latency_p95_ms:.2f} ms, "
        f"throughp[?2004l
/Users/tienlishen/Library/Python/3.9/lib/python/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(
usage: bench_m2_coreml_vs_cpu.py [-h] --model-name MODEL_NAME [--onnx-path ONNX_PATH]
                                 [--verbose-ort]
bench_m2_coreml_vs_cpu.py: error: the following arguments are required: --model-name
[1m[7m%[27m[1m[0m                                                                                                  [0m[27m[24m[Jtienlishen@Mac coreml-ep-performance-study % [K[?2004hpplot_core       p  ppython3 scripts[1m/[0m[0m/plot_coreml_vs_cpu.py[1m [0m[0m [?2004l
[WARN] No CSV files found in results/csv matching m2_coreml_vs_cpu_*.csv
[1m[7m%[27m[1m[0m                                                                                                  [0m[27m[24m[Jtienlishen@Mac coreml-ep-performance-study % [K[?2004hpython3 scripts/plot_coreml_vs_cpu.py[?2004l
[WARN] No CSV files found in results/csv matching m2_coreml_vs_cpu_*.csv
[1m[7m%[27m[1m[0m                                                                                                  [0m[27m[24m[Jtienlishen@Mac coreml-ep-performance-study % [K[?2004hpython3 scripts/plot_coreml_vs_cpu.py[?2004l
[WARN] No CSV files found in results/csv matching m2_coreml_vs_cpu_*.csv
[1m[7m%[27m[1m[0m                                                                                                  [0m[27m[24m[Jtienlishen@Mac coreml-ep-performance-study % [K[?2004hpython3 scripts/plot_coreml_vs_cpu.py[?2004l
[INFO] Found 3 CSV file(s) to plot in results/csv
[INFO] Processing CSV: results/csv/m2_coreml_vs_cpu_bert-base-uncased.csv
[INFO] Saved combined plot to results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_bert-base-uncased_combined.png
[INFO] Processing CSV: results/csv/m2_coreml_vs_cpu_distilbert-base-uncased.csv
[INFO] Saved combined plot to results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_distilbert-base-uncased_combined.png
[INFO] Processing CSV: results/csv/m2_coreml_vs_cpu_tiny-systems-bert.csv
[INFO] Saved combined plot to results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_tiny-systems-bert_combined.png
[1m[7m%[27m[1m[0m                                                                                                  [0m[27m[24m[Jtienlishen@Mac coreml-ep-performance-study % [K[?2004hpython3 scripts/plot_coreml_vs_cpu.py[?2004l
[INFO] Found 3 CSV file(s) to plot in results/csv
[INFO] Processing CSV: results/csv/m2_coreml_vs_cpu_bert-base-uncased.csv
[INFO] Saved combined plot to results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_bert-base-uncased_combined.png
[INFO] Processing CSV: results/csv/m2_coreml_vs_cpu_distilbert-base-uncased.csv
[INFO] Saved combined plot to results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_distilbert-base-uncased_combined.png
[INFO] Processing CSV: results/csv/m2_coreml_vs_cpu_tiny-systems-bert.csv
[INFO] Saved combined plot to results/plots/m2_coreml_vs_cpu/m2_coreml_vs_cpu_tiny-systems-bert_combined.png
[1m[7m%[27m[1m[0m                                                                                                  [0m[27m[24m[Jtienlishen@Mac coreml-ep-performance-study % [K[?2004h