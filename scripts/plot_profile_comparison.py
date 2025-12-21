#!/usr/bin/env python3
"""
Generate a bar graph comparing latencies across tiny-systems-bert configurations.
"""

import re
import matplotlib.pyplot as plt
from pathlib import Path


def parse_profile_summary(file_path):
    """Extract key metrics from a profile summary file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract total duration
    duration_match = re.search(r'Total \(dur sum\):\s+([\d.]+)\s+ms', content)
    total_duration = float(duration_match.group(1)) if duration_match else None
    
    # Count unique CoreML partitions from the profile
    # Look for patterns like CoreMLExecutionProvider_CoreML_<numbers>_<id>_<id>_kernel_time
    partition_pattern = r'CoreMLExecutionProvider_CoreML_\d+_(\d+)_\1_kernel_time'
    partition_ids = set(re.findall(partition_pattern, content))
    partitions = len(partition_ids) if partition_ids else None
    
    # Extract node count
    node_match = re.search(r'number of nodes in the graph:\s+(\d+)', content)
    total_nodes = int(node_match.group(1)) if node_match else None
    
    # Extract supported nodes
    supported_match = re.search(r'number of nodes supported by CoreML:\s+(\d+)', content)
    coreml_nodes = int(supported_match.group(1)) if supported_match else None
    
    # Extract executor time (SequentialExecutor::Execute)
    executor_match = re.search(r'SequentialExecutor::Execute\s+[\d.]+\s+ms\s+\((\d+)\s+events\)', content)
    
    # Try to find mean latency from benchmark results (if available)
    latency_match = re.search(r'Mean latency:\s+([\d.]+)\s+ms', content)
    mean_latency = float(latency_match.group(1)) if latency_match else None
    
    # If no mean latency, estimate from session time
    if mean_latency is None and duration_match:
        # Use total duration / number of runs as rough estimate
        mean_latency = total_duration / 105  # Assuming 105 runs
    
    return {
        'total_duration_ms': total_duration,
        'mean_latency_ms': mean_latency,
        'partitions': partitions,
        'total_nodes': total_nodes,
        'coreml_nodes': coreml_nodes
    }


def main():
    profiles_dir = Path('results/txt')
    
    # Define the profiles and their display names
    profiles = [
        ('tiny-systems-bert_fp32_dynamic_gelu_profile_summary.txt', 
         'FP32 Dynamic\nGELU'),
        ('tiny-systems-bert_fp32_static_b1_s128_gelu_profile_summary.txt',
         'FP32 Static\nGELU'),
        ('tiny-systems-bert_fp32_static_b1_s128_fast-gelu_profile_summary.txt',
         'FP32 Static\nFastGELU'),
        ('tiny-systems-bert_fp16_static_b1_s128_fast-gelu_profile_summary.txt',
         'FP16 Static\nFastGELU'),
    ]
    
    # Parse all profiles
    data = []
    labels = []
    partitions = []
    
    for filename, label in profiles:
        file_path = profiles_dir / filename
        if file_path.exists():
            metrics = parse_profile_summary(file_path)
            if metrics['mean_latency_ms']:
                data.append(metrics['mean_latency_ms'])
                labels.append(label)
                partitions.append(metrics['partitions'])
                print(f"{label}: {metrics['mean_latency_ms']:.2f} ms, {metrics['partitions']} partitions")
        else:
            print(f"Warning: {filename} not found")
    
    if not data:
        print("Error: No valid profiles found")
        return
    
    # Create the bar graph
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(range(len(data)), data, color=colors[:len(data)], alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Customize the plot
    ax.set_ylabel('Mean Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Tiny-Systems-BERT Performance Comparison\nAcross Configuration Variants', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, max(data) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val, parts) in enumerate(zip(bars, data, partitions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{val:.2f} ms\n({parts} partitions)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path('results/plots/tiny-systems-bert_latency_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    
    plt.close()


if __name__ == '__main__':
    main()
