import time
import os
import json
import numpy as np
import onnxruntime as ort

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", f"{MODEL_NAME}.onnx")
BATCH_SIZE = 1
SEQ_LEN = 128
WARMUP = 10
RUNS = 100


def make_dummy_input(input_def):
    # Build concrete shape from dynamic dims
    shape = []
    for d in input_def.shape:
        if isinstance(d, int):
            shape.append(d)
        else:
            # assume first dynamic dim is batch, second is sequence length
            if len(shape) == 0:
                shape.append(BATCH_SIZE)
            else:
                shape.append(SEQ_LEN)

    name = input_def.name.lower()

    # BERT-style inputs
    if "input_ids" in name:
        # token ids in vocab range (roughly)
        return np.random.randint(0, 30000, size=shape, dtype=np.int64)
    elif "token_type" in name:
        # segment ids: 0 or 1
        return np.random.randint(0, 2, size=shape, dtype=np.int64)
    elif "attention_mask" in name or "mask" in name:
        # 1 for real tokens, 0 for padding (we just use 1s)
        return np.ones(shape, dtype=np.int64)
    else:
        # fallback for any float tensors
        return np.random.randn(*shape).astype(np.float32)


def main():
    print(f"Loading ONNX model from {MODEL_PATH}")
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    inputs = session.get_inputs()
    dummy_feed = {inp.name: make_dummy_input(inp) for inp in inputs}

    # Warmup
    for _ in range(WARMUP):
        session.run(None, dummy_feed)

    # Timed runs
    latencies = []
    for _ in range(RUNS):
        start = time.perf_counter()
        session.run(None, dummy_feed)
        end = time.perf_counter()
        latencies.append(end - start)

    lat = np.array(latencies)
    mean = lat.mean()
    throughput = RUNS / lat.sum()

    print(f"Runs: {RUNS}")
    print(f"Mean latency: {mean * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} inferences/sec")
    summary = {
        "model": MODEL_NAME,
        "backend": "cpu",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "runs": RUNS,
        "mean_latency_ms": mean * 1000,
        "throughput": throughput,
    }

    os.makedirs("results/raw", exist_ok=True)
    out_path = os.path.join("results", "raw", f"{MODEL_NAME}_cpu_bs{BATCH_SIZE}_seq{SEQ_LEN}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    BATCH_SIZE = args.batch  # override the constant
    main()
