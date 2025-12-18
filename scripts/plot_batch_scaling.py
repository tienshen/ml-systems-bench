#!/usr/bin/env python3
"""
Plot batch size scaling comparison between CPU and CoreML execution providers.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from benchmarks
batch_sizes = [1, 2, 4, 8, 16, 32]

# CPU results
cpu_latency = [1.14, 1.72, 3.11, 5.27, 10.95, 23.34]  # ms
cpu_throughput = [875.68, 1161.96, 1285.63, 1517.19, 1461.48, 1371.18]  # inferences/sec

# CoreML results
coreml_latency = [1.16, 1.69, 3.13, 5.23, 10.90, 24.83]  # ms
coreml_throughput = [861.91, 1182.59, 1276.44, 1530.26, 1467.77, 1288.68]  # inferences/sec

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Latency vs Batch Size
ax1.plot(batch_sizes, cpu_latency, 'o-', label='CPU', linewidth=2, markersize=8)
ax1.plot(batch_sizes, coreml_latency, 's-', label='CoreML', linewidth=2, markersize=8)
ax1.set_xlabel('Batch Size', fontsize=12)
ax1.set_ylabel('Mean Latency (ms)', fontsize=12)
ax1.set_title('Latency vs Batch Size\n(tiny-systems-bert, FastGELU, FP16)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_xticks(batch_sizes)
ax1.set_xticklabels(batch_sizes)

# Add latency per sample annotations
for i, bs in enumerate(batch_sizes):
    cpu_per_sample = cpu_latency[i] / bs
    coreml_per_sample = coreml_latency[i] / bs
    if i in [0, 3, 5]:  # Annotate a few points to avoid clutter
        ax1.annotate(f'{cpu_per_sample:.2f}ms/sample', 
                     xy=(bs, cpu_latency[i]), 
                     xytext=(10, -5), 
                     textcoords='offset points', 
                     fontsize=8, 
                     color='C0', 
                     alpha=0.7)

# Plot 2: Throughput vs Batch Size
ax2.plot(batch_sizes, cpu_throughput, 'o-', label='CPU', linewidth=2, markersize=8)
ax2.plot(batch_sizes, coreml_throughput, 's-', label='CoreML', linewidth=2, markersize=8)
ax2.set_xlabel('Batch Size', fontsize=12)
ax2.set_ylabel('Throughput (inferences/sec)', fontsize=12)
ax2.set_title('Throughput vs Batch Size\n(tiny-systems-bert, FastGELU, FP16)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)
ax2.set_xticks(batch_sizes)
ax2.set_xticklabels(batch_sizes)

# Mark optimal batch size
cpu_max_idx = np.argmax(cpu_throughput)
coreml_max_idx = np.argmax(coreml_throughput)
ax2.scatter([batch_sizes[cpu_max_idx]], [cpu_throughput[cpu_max_idx]], 
            s=200, facecolors='none', edgecolors='C0', linewidth=2, zorder=5)
ax2.scatter([batch_sizes[coreml_max_idx]], [coreml_throughput[coreml_max_idx]], 
            s=200, facecolors='none', edgecolors='C1', linewidth=2, zorder=5)

fig.suptitle('Batch Scaling on Apple M2 — tiny-systems-bert • FastGELU • FP16', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('results/plots/batch_scaling_cpu_vs_coreml.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved to results/plots/batch_scaling_cpu_vs_coreml.png")

# Print summary statistics
print("\n=== Batch Scaling Analysis ===")
print(f"\nOptimal Batch Size:")
print(f"  CPU: batch={batch_sizes[cpu_max_idx]} → {cpu_throughput[cpu_max_idx]:.0f} inf/sec")
print(f"  CoreML: batch={batch_sizes[coreml_max_idx]} → {coreml_throughput[coreml_max_idx]:.0f} inf/sec")

print(f"\nLatency Per Sample at Different Batch Sizes:")
for i, bs in enumerate(batch_sizes):
    cpu_per = cpu_latency[i] / bs
    coreml_per = coreml_latency[i] / bs
    speedup = cpu_per / coreml_per
    print(f"  batch={bs:2d}: CPU={cpu_per:5.2f}ms/sample, CoreML={coreml_per:5.2f}ms/sample (speedup: {speedup:.2f}x)")

print(f"\nThroughput Comparison:")
for i, bs in enumerate(batch_sizes):
    ratio = coreml_throughput[i] / cpu_throughput[i]
    winner = "CoreML" if ratio > 1 else "CPU"
    print(f"  batch={bs:2d}: CPU={cpu_throughput[i]:7.1f}, CoreML={coreml_throughput[i]:7.1f} → {winner} by {abs(ratio-1)*100:4.1f}%")
