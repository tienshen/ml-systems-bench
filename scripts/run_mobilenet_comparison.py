#!/usr/bin/env python3
"""
Run MobileNet benchmarks and generate comparison plot with live data.
This script pipelines data from run_mac_bench directly to plotting.
The default setting is to generate plot from existing CSV for easier plot customization/revision,
otherwise set --rerun to run fresh benchmarks.
"""

import sys
from pathlib import Path
import json
import csv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the benchmark runner
import run_mac_bench
import onnxruntime as ort


def analyze_coreml_usage(profile_path):
    """Analyze CoreML vs CPU usage from profile JSON."""
    if profile_path is None or not Path(profile_path).exists():
        return None
    
    with open(profile_path) as f:
        data = json.load(f)
    
    kernels = [e for e in data if e.get('cat') == 'Node' and 'kernel_time' in e.get('name', '')]
    coreml_ops = [e for e in kernels if 'CoreML' in e['name']]
    cpu_ops = [e for e in kernels if 'CoreML' not in e['name']]
    
    coreml_time = sum(e['dur'] for e in coreml_ops)
    cpu_time = sum(e['dur'] for e in cpu_ops)
    total = coreml_time + cpu_time
    
    if total == 0:
        return 0.0
    
    return (coreml_time / total) * 100


def run_mobilenet_comparison(model_base="mobilenet_v2_b1_h224_w224", batch_size=1, profile_dir="profiles", cooldown_seconds=3):
    """
    Run all four configurations and return results.
    Includes cooldown periods between benchmarks to let the system stabilize.
    """
    import time
    
    print("="*70)
    print("Running MobileNet FP16 vs FP32 CoreML Comparison")
    print("="*70)
    
    results = []
    cpu_providers = run_mac_bench.make_providers("cpu")
    coreml_providers = run_mac_bench.make_providers("coreml")
    
    # Configuration 1: FP16 with CPU
    print("\n[1/4] FP16 with CPUExecutionProvider...")
    r1 = run_mac_bench.run_bench(
        model_name=f"{model_base}_fp16",
        providers=cpu_providers,
        batch_size=batch_size,
        seq_len=128,  # Not used for vision models
        profile_dir=profile_dir,
        verbose=False
    )
    r1['config_name'] = 'FP16 with CPU'
    r1['coreml_usage_pct'] = None  # N/A for CPU-only
    results.append(r1)
    
    # Cooldown
    print(f"  Cooling down for {cooldown_seconds}s...")
    time.sleep(cooldown_seconds)
    
    # Configuration 2: FP32 with CPU
    print("\n[2/4] FP32 with CPUExecutionProvider...")
    r2 = run_mac_bench.run_bench(
        model_name=f"{model_base}_fp32",
        providers=cpu_providers,
        batch_size=batch_size,
        seq_len=128,
        profile_dir=profile_dir,
        verbose=False
    )
    r2['config_name'] = 'FP32 with CPU'
    r2['coreml_usage_pct'] = None  # N/A for CPU-only
    results.append(r2)
    
    # Cooldown
    print(f"  Cooling down for {cooldown_seconds}s...")
    time.sleep(cooldown_seconds)
    
    # Configuration 3: FP16 with CoreML (will fail to partition)
    print("\n[3/4] FP16 with CoreMLExecutionProvider...")
    r3 = run_mac_bench.run_bench(
        model_name=f"{model_base}_fp16",
        providers=coreml_providers,
        batch_size=batch_size,
        seq_len=128,
        profile_dir=profile_dir,
        verbose=False
    )
    r3['config_name'] = 'FP16 with CoreML'
    r3['coreml_usage_pct'] = analyze_coreml_usage(r3['profile_path'])
    results.append(r3)
    
    # Cooldown
    print(f"  Cooling down for {cooldown_seconds}s...")
    time.sleep(cooldown_seconds)
    
    # Configuration 4: FP32 with CoreML (should partition successfully)
    print("\n[4/4] FP32 with CoreMLExecutionProvider...")
    r4 = run_mac_bench.run_bench(
        model_name=f"{model_base}_fp32",
        providers=coreml_providers,
        batch_size=batch_size,
        seq_len=128,
        profile_dir=profile_dir,
        verbose=False
    )
    r4['config_name'] = 'FP32 with CoreML'
    r4['coreml_usage_pct'] = analyze_coreml_usage(r4['profile_path'])
    results.append(r4)
    
    return results


def plot_comparison(results, output_path="results/plots/mobilenet_coreml_cpu_comparison.png"):
    """Generate comparison plot from benchmark results."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract data
    configs = [r['config_name'] for r in results]
    latencies = [r['mean_latency_ms'] for r in results]
    throughputs = [r['throughput_per_sec'] for r in results]
    coreml_usage = [r.get('coreml_usage_pct', 0) if r.get('coreml_usage_pct') is not None else 0 
                    for r in results]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Colors: FP16+CPU (orange), FP32+CPU (blue), FP16+CoreML (red/failed), FP32+CoreML (teal/success)
    colors = ['#FFA500', '#4A90E2', '#FF6B6B', '#4ECDC4']
    
    # Plot 1: Latency (lower is better)
    bars1 = ax1.bar(range(len(configs)), latencies, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Mean Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency Comparison\n(Lower is Better)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars1, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.2f}ms',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add speedup annotation (FP16+CPU baseline vs FP32+CoreML)
    if len(latencies) >= 4:
        speedup = latencies[0] / latencies[3]
        ax1.annotate('', xy=(3, latencies[3]), xytext=(0, latencies[0]),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax1.text(1.5, (latencies[0] + latencies[3])/2, f'{speedup:.1f}x faster!', 
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Throughput (higher is better)
    bars2 = ax2.bar(range(len(configs)), throughputs, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Throughput (inferences/sec)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput Comparison\n(Higher is Better)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars2, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.0f}/s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add throughput increase annotation (FP16+CPU baseline vs FP32+CoreML)
    if len(throughputs) >= 4:
        increase = throughputs[3] / throughputs[0]
        ax2.annotate('', xy=(3, throughputs[3]), xytext=(0, throughputs[0]),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax2.text(1.5, (throughputs[0] + throughputs[3])/2, f'{increase:.1f}x more!', 
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    # Plot 3: CoreML Partition Usage
    bars3 = ax3.bar(range(len(configs)), coreml_usage, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('CoreML Partition Usage (%)', fontsize=12, fontweight='bold')
    ax3.set_title('CoreML Graph Partitioning\n(Higher is Better)', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, fontsize=10)
    ax3.set_ylim(0, 110)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, coreml_usage)):
        height = bar.get_height()
        if results[i].get('coreml_usage_pct') is None:
            label = 'N/A\n(CPU-only)'
        elif val < 10:
            label = f'{val:.0f}%\n(CPU fallback)'
        else:
            label = f'{val:.0f}%\n(CoreML partition)'
        ax3.text(bar.get_x() + bar.get_width()/2., height + 3,
                 label,
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Critical Performance Pitfall: Pre-converted FP16 ONNX Models Fail with CoreML EP\nMobileNetV2 on Apple M2', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    # Add explanation text at bottom
    fig.text(0.5, -0.12, 
             'Key Finding: Pre-converted FP16 ONNX models prevent CoreML graph partitioning, causing CPU fallback!\n'
             'FP32 ONNX let CoreML EP handle the conversion internally for proper partitioning.',
             ha='center', fontsize=11, 
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    
    return output_path


def load_results_from_csv(csv_path):
    """Load existing benchmark results from CSV file."""
    if not Path(csv_path).exists():
        return None
    
    results = []
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Reconstruct result dict from CSV
            result = {
                'model_name': row['model_name'],
                'config_name': row['configuration'],
                'batch_size': int(row['batch_size']),
                'providers': row['providers'].split(','),
                'mean_latency_ms': float(row['mean_latency_ms']),
                'p50_latency_ms': float(row['p50_latency_ms']),
                'p90_latency_ms': float(row['p90_latency_ms']),
                'p99_latency_ms': float(row['p99_latency_ms']),
                'throughput_per_sec': float(row['throughput_per_sec']),
                'coreml_usage_pct': float(row['coreml_usage_pct']) if row['coreml_usage_pct'] != 'N/A' else None,
                'profile_path': row['profile_path']
            }
            results.append(result)
    
    return results if results else None


def export_to_csv(results, output_path="results/csv/mobilenet_comparison.csv"):
    """Export benchmark results to CSV file."""
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define CSV columns
    fieldnames = [
        'timestamp',
        'model_name',
        'configuration',
        'batch_size',
        'providers',
        'mean_latency_ms',
        'p50_latency_ms',
        'p90_latency_ms',
        'p99_latency_ms',
        'throughput_per_sec',
        'coreml_usage_pct',
        'profile_path'
    ]
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'timestamp': timestamp,
                'model_name': r['model_name'],
                'configuration': r['config_name'].replace('\n', ' '),
                'batch_size': r['batch_size'],
                'providers': ','.join([p if isinstance(p, str) else p[0] for p in r['providers']]),
                'mean_latency_ms': f"{r['mean_latency_ms']:.4f}",
                'p50_latency_ms': f"{r['p50_latency_ms']:.4f}",
                'p90_latency_ms': f"{r['p90_latency_ms']:.4f}",
                'p99_latency_ms': f"{r['p99_latency_ms']:.4f}",
                'throughput_per_sec': f"{r['throughput_per_sec']:.2f}",
                'coreml_usage_pct': f"{r['coreml_usage_pct']:.2f}" if r.get('coreml_usage_pct') is not None else 'N/A',
                'profile_path': r.get('profile_path', '')
            })
    
    print(f"\n✓ Results exported to {output_path}")
    return output_path


def print_summary(results):
    """Print detailed comparison summary."""
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"\n{results[0]['model_name'].split('_')[0].upper()} on Apple M2 - Batch Size {results[0]['batch_size']}:")
    print("-" * 70)
    print(f"{'Configuration':<40} {'Latency':<12} {'Throughput':<15} {'CoreML %'}")
    print("-" * 70)
    
    for r in results:
        config_clean = r['config_name'].replace('\n', ' ')
        latency = f"{r['mean_latency_ms']:.2f}ms"
        throughput = f"{r['throughput_per_sec']:.1f}/sec"
        
        if r.get('coreml_usage_pct') is None:
            coreml_pct = "N/A"
        else:
            coreml_pct = f"{r['coreml_usage_pct']:.0f}%"
        
        print(f"{config_clean:<40} {latency:<12} {throughput:<15} {coreml_pct}")
    
    print("-" * 70)
    
    # Calculate key metrics (assuming order: FP16+CPU, FP32+CPU, FP16+CoreML, FP32+CoreML)
    if len(results) >= 4:
        fp16_cpu = results[0]
        fp32_cpu = results[1]
        fp16_coreml = results[2]
        fp32_coreml = results[3]
        
        # Main comparison: FP32+CoreML vs FP16+CPU baseline
        speedup_best = fp16_cpu['mean_latency_ms'] / fp32_coreml['mean_latency_ms']
        throughput_gain_best = fp32_coreml['throughput_per_sec'] / fp16_cpu['throughput_per_sec']
        
        # FP16 precision impact on CPU
        fp_precision_cpu = fp16_cpu['mean_latency_ms'] / fp32_cpu['mean_latency_ms']
        
        # CoreML partitioning failure impact
        partitioning_failure = fp16_coreml['mean_latency_ms'] / fp32_coreml['mean_latency_ms']
        
        print(f"\nKey Findings:")
        print(f"  • FP32 with CoreML achieves {speedup_best:.1f}x speed up over CPU baseline")
        print(f"  • FP32 with CoreML achieves {throughput_gain_best:.1f}x higher throughput")
        print(f"  • FP16 partitioning failure causes {partitioning_failure:.1f}x slowdown vs FP32 in CoreML")
        print(f"  • FP16 vs FP32 precision on CPU: {fp_precision_cpu:.2f}x difference")
    
    print("\n" + "="*70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run MobileNet FP16 vs FP32 CoreML comparison with pipelined data"
    )
    parser.add_argument(
        "--model-base",
        default="mobilenet_v2_b1_h224_w224",
        help="Base model name (without _fp16/_fp32 suffix)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--profile-dir",
        default="profiles",
        help="Directory to save profiling data"
    )
    parser.add_argument(
        "--output",
        default="results/plots/mobilenet_fp16_vs_fp32_coreml_vs_cpu.png",
        help="Output plot path"
    )
    parser.add_argument(
        "--csv",
        default="results/csv/mobilenet_comparison.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerun experiments even if CSV results already exist"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=3,
        help="Cooldown time in seconds between benchmarks (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Check if CSV already exists and load cached results
    results = None
    if not args.rerun:
        results = load_results_from_csv(args.csv)
        if results:
            print(f"✓ Found existing results in {args.csv}")
            print("  Using cached data. Run with --rerun to force new benchmarks.")
    
    # Run benchmarks if no cached results or rerun requested
    if results is None:
        results = run_mobilenet_comparison(
            model_base=args.model_base,
            batch_size=args.batch,
            profile_dir=args.profile_dir,
            cooldown_seconds=args.cooldown
        )
        
        # Export to CSV
        export_to_csv(results, output_path=args.csv)
    
    # Generate plot from data (cached or fresh)
    plot_comparison(results, output_path=args.output)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
