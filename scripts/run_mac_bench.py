import os
import time
import socket
import argparse
import numpy as np
import onnxruntime as ort

RUNS = 100

def make_dummy_input(input_defs, batch_size, seq_len):
    feeds = {}
    for inp in input_defs:
        name = inp.name
        shape = []
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            else:
                # assume [batch, seq]
                if len(shape) == 0:
                    shape.append(batch_size)
                else:
                    shape.append(seq_len)

        lname = name.lower()
        if "input_ids" in lname:
            arr = np.random.randint(0, 30000, size=shape, dtype=np.int64)
        elif "token_type" in lname:
            arr = np.zeros(shape, dtype=np.int64)
        elif "attention" in lname:
            arr = np.ones(shape, dtype=np.int64)
        else:
            arr = np.random.randint(0, 10000, size=shape, dtype=np.int64)

        feeds[name] = arr
    return feeds

def run_bench(model_name, provider, batch_size, seq_len):
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(models_dir, f"{model_name}.onnx")
    print(f"\n=== {model_name} on {provider}, batch={batch_size}, seq={seq_len} ===")
    print(f"Loading ONNX model from {model_path}")

    sess = ort.InferenceSession(model_path, providers=[provider])
    inputs = sess.get_inputs()
    feed = make_dummy_input(inputs, batch_size, seq_len)

    # warmup
    for _ in range(5):
        sess.run(None, feed)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times.append(t1 - t0)

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
    args = parser.parse_args()

    providers_to_test = ["CPUExecutionProvider", "CoreMLExecutionProvider"]

    for ep in providers_to_test:
        if ep in ort.get_available_providers():
            run_bench(args.model, ep, args.batch, args.seq_len)
        else:
            print(f"Skipping {ep} (not available)")
