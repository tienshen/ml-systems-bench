#!/usr/bin/env python3
"""
Benchmark a Core ML model (.mlpackage or .mlmodel) for latency and throughput.
Supports static and dynamic batch sizes, and FP16/FP32 settings.
"""
import argparse
import time
import numpy as np
import coremltools as ct
from pathlib import Path


def run_bench(model_path, batch_size=1, num_warmup=5, num_runs=50, dtype='fp32', dynamic_batch=False, compute_unit='all'):
    # Set compute unit
    config = ct.models.MLModelConfiguration()
    if compute_unit == 'cpu':
        config.compute_units = ct.ComputeUnit.CPU_ONLY
    elif compute_unit == 'gpu':
        config.compute_units = ct.ComputeUnit.CPU_AND_GPU
    elif compute_unit == 'neural':
        config.compute_units = ct.ComputeUnit.CPU_AND_NE
    else:
        config.compute_units = ct.ComputeUnit.ALL
    # Load model
    model = ct.models.MLModel(model_path, configuration=config)
    spec = model.get_spec()
    # Get input name and shape from spec
    if len(spec.description.input) == 0:
        raise ValueError("Model has no inputs!")
    input_name = spec.description.input[0].name
    input_type = spec.description.input[0].type
    if input_type.WhichOneof('Type') == 'multiArrayType':
        input_shape = [d for d in input_type.multiArrayType.shape]
    elif input_type.WhichOneof('Type') == 'imageType':
        # For image input, shape is [batch, channels, height, width] (assume batch=1 if not dynamic)
        c = input_type.imageType.colorSpace
        if c == 10:  # RGB
            channels = 3
        else:
            channels = 1
        input_shape = [batch_size, channels, input_type.imageType.height, input_type.imageType.width]
    else:
        raise ValueError("Unsupported input type for benchmarking.")
    # If dynamic batch, override batch dim
    if dynamic_batch:
        input_shape[0] = batch_size
    # Set dtype
    np_dtype = np.float16 if dtype == 'fp16' else np.float32
    # Generate random input
    input_data = np.random.randn(*input_shape).astype(np_dtype)
    # Warmup
    for _ in range(num_warmup):
        _ = model.predict({input_name: input_data})
    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict({input_name: input_data})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    mean_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    throughput = (batch_size * num_runs) / (np.sum(latencies) / 1000)
    return {
        'model_path': str(model_path),
        'batch_size': batch_size,
        'dtype': dtype,
        'dynamic_batch': dynamic_batch,
        'mean_latency_ms': mean_latency,
        'p50_latency_ms': p50,
        'p90_latency_ms': p90,
        'p99_latency_ms': p99,
        'throughput_per_sec': throughput,
        'num_runs': num_runs
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark a Core ML model for latency and throughput.')
    parser.add_argument('--model', type=str, required=True, help='Path to .mlpackage or .mlmodel')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for input')
    parser.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32', help='Input data type')
    parser.add_argument('--dynamic-batch', action='store_true', help='Use dynamic batch size (if model supports)')
    parser.add_argument('--num-warmup', type=int, default=5, help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of timed runs')
    parser.add_argument('--compute-unit', choices=['all', 'cpu', 'gpu', 'neural'], default='all', help='Core ML compute unit to use')
    args = parser.parse_args()

    result = run_bench(
        model_path=args.model,
        batch_size=args.batch_size,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        dtype=args.dtype,
        dynamic_batch=args.dynamic_batch,
        compute_unit=args.compute_unit
    )
    print("\nBenchmark Result:")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
