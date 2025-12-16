import os
import time
import argparse
import numpy as np
import onnxruntime as ort
from typing import Dict, Any
from transformers import AutoTokenizer
from pathlib import Path
import re


RUNS = 2
WARMUP = 1

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

def infer_tokenizer_from_onnx_path(onnx_path: str) -> str:
    stem = Path(onnx_path).stem  # remove .onnx
    # strip suffixes like _b1_s128, _fastgelu, _static, etc.
    stem = re.sub(r"_b\d+_s\d+.*$", "", stem)
    stem = re.sub(r"_s\d+_b\d+.*$", "", stem)
    stem = re.sub(r"_fastgelu.*$", "", stem)
    stem = re.sub(r"_static.*$", "", stem)
    return stem

def make_providers(ep_mode: str):
    """
    ep_mode:
      - "cpu"         -> CPU only
      - "coreml"      -> CoreML only (may fail if any op unsupported)
      - "coreml_cpu"  -> CoreML preferred + CPU fallback (recommended for diagnosis)
    """
    if ep_mode == "cpu":
        return ["CPUExecutionProvider"]
    if ep_mode == "coreml":
        return ["CoreMLExecutionProvider"]
    if ep_mode == "coreml_cpu":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"Unknown --ep {ep_mode}")

def run_bench(model_name, providers, batch_size, seq_len, profile_dir=None, verbose=False,
             onnx_path=None, tokenizer_name=None):
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(models_dir, f"{model_name}.onnx")

    print(f"\n=== {model_name} on {providers}, batch={batch_size}, seq={seq_len} ===")
    print(f"Loading ONNX model from {model_path}")

    so = ort.SessionOptions()
    if verbose:
        so.log_severity_level = 0   # VERBOSE
        so.log_verbosity_level = 1

    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        so.enable_profiling = True
        # Newer ORT supports this; older versions might not.
        # Safe to try; if it errors, remove these two lines.
        try:
            so.profile_file_prefix = os.path.join(
                profile_dir,
                f"{model_name}_b{batch_size}_s{seq_len}_" + "_".join([p if isinstance(p, str) else p[0] for p in providers])
            )
        except Exception:
            pass

    # providers can be list[str] or list[("EP", {opts})]; we keep list[str] here
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    for i in sess.get_inputs():
        print(i.name, i.shape, i.type)
    print("Session providers (resolved):", sess.get_providers())

    tokenizer = AutoTokenizer.from_pretrained("tiny-systems-bert", use_fast=True) #model_name)
    feed = make_dummy_inputs(tokenizer, seq_len, batch_size)

    # warmup
    for _ in range(WARMUP):
        sess.run(None, feed)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    if profile_dir is not None:
        prof_path = sess.end_profiling()
        print("ORT profile written to:", prof_path)

    times = np.array(times)
    mean = times.mean() * 1000
    p50 = np.percentile(times, 50) * 1000
    p90 = np.percentile(times, 90) * 1000
    p99 = np.percentile(times, 99) * 1000
    throughput = RUNS * batch_size / times.sum()

    print(f"Mean latency: {mean:.2f} ms")
    print(f"p50: {p50:.2f} ms  p90: {p90:.2f} ms  p99: {p99:.2f} ms")
    print(f"Throughput: {throughput:.2f} inferences/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--ep", type=str, default="coreml_cpu", choices=["cpu", "coreml", "coreml_cpu"])
    parser.add_argument("--profile-dir", type=str, default=None, help="If set, enable ORT profiling and write JSON here")
    parser.add_argument("--verbose", action="store_true", help="Enable ORT verbose logging")
    args = parser.parse_args()

    requested = make_providers(args.ep)
    print("Requested providers:", requested)

    # availability guard
    for p in requested:
        if p not in ort.get_available_providers():
            print(f"Requested provider not available: {p}")
            print("Available providers:", ort.get_available_providers())
            raise SystemExit(2)

    run_bench(args.model, requested, args.batch, args.seq_len, profile_dir=args.profile_dir, verbose=args.verbose)
