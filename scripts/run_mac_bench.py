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
    """
    Infer the HuggingFace tokenizer name from ONNX model filename.
    
    Examples:
        tiny-systems-bert_b1_s128_fast-gelu_fp16 -> tiny-systems-bert
        bert-base-uncased_b1_s128_gelu_fp32 -> bert-base-uncased
        distilbert-base-uncased -> distilbert-base-uncased
    """
    stem = Path(onnx_path).stem  # remove .onnx
    # Strip common suffixes: batch/seq dims, activation type, precision
    stem = re.sub(r"_b\d+_s\d+.*$", "", stem)  # _b1_s128...
    stem = re.sub(r"_s\d+_b\d+.*$", "", stem)  # _s128_b1...
    stem = re.sub(r"_(fast-)?gelu.*$", "", stem)  # _gelu, _fast-gelu
    stem = re.sub(r"_fp(16|32).*$", "", stem)  # _fp16, _fp32 (explicit precision)
    stem = re.sub(r"_static.*$", "", stem)  # _static
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
    
    # Enable graph optimization
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
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

    # Infer tokenizer name from model name
    # Strip suffixes like _b1_s128, _fast-gelu, _fp16, etc. to get base model name
    if tokenizer_name is None:
        tokenizer_name = infer_tokenizer_from_onnx_path(model_name)
    
    print(f"Using tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
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
        
        # Rename to remove timestamp and use consistent naming
        import shutil
        import subprocess
        import sys
        
        new_prof_path = os.path.join(
            profile_dir,
            f"{model_name}_b{batch_size}_s{seq_len}_" + "_".join([p if isinstance(p, str) else p[0] for p in providers]) + ".json"
        )
        shutil.move(prof_path, new_prof_path)
        print("ORT profile written to:", new_prof_path)
        
        # Automatically generate summary
        summary_path = new_prof_path.replace(".json", "_summary.txt")
        summarizer_script = os.path.join(os.path.dirname(__file__), "summarize_ort_profile.py")
        
        try:
            subprocess.run(
                [sys.executable, summarizer_script, new_prof_path, "--top", "50", "--output", summary_path],
                check=True,
                capture_output=True,
                text=True
            )
            print("Profile summary written to:", summary_path)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate profile summary: {e}")
            if e.stderr:
                print(f"  Error: {e.stderr}")

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
    parser.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer name (default: inferred from model name)")
    args = parser.parse_args()

    requested = make_providers(args.ep)
    print("Requested providers:", requested)

    # availability guard
    for p in requested:
        if p not in ort.get_available_providers():
            print(f"Requested provider not available: {p}")
            print("Available providers:", ort.get_available_providers())
            raise SystemExit(2)

    run_bench(args.model, requested, args.batch, args.seq_len, 
              profile_dir=args.profile_dir, verbose=args.verbose, tokenizer_name=args.tokenizer)
