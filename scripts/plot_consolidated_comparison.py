#!/usr/bin/env python3
"""
Consolidate all m2_coreml_vs_cpu CSV results into a single 2-subplot figure.
Shows latency and throughput comparison across all models (BERT, DistilBERT, Tiny-Systems-BERT).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_all_csvs():
    """Load all m2_coreml_vs_cpu CSV files."""
    csv_dir = Path('results/csv')
    csv_files = sorted(csv_dir.glob('m2_coreml_vs_cpu_*.csv'))
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract model name from filename
        model_name = csv_file.stem.replace('m2_coreml_vs_cpu_', '')
        df['model'] = model_name
        all_data.append(df)
        print(f"Loaded {csv_file.name}: {len(df)} rows")
    
    if not all_data:
        raise ValueError("No CSV files found")
    
    return pd.concat(all_data, ignore_index=True)


def create_consolidated_plot(df):
    """Create 2-subplot figure with latency and throughput comparisons."""
    
    # Filter for seq_len=128, batch_size=1 for fair comparison
    df_filtered = df[(df['seq_len'] == 128) & (df['batch_size'] == 1)].copy()
    
    # Create readable labels
    model_labels = {
        'bert-base-uncased': 'BERT-base',
        'distilbert-base-uncased': 'DistilBERT',
        'tiny-systems-bert': 'Tiny-BERT'
    }
    df_filtered['model_label'] = df_filtered['model'].map(model_labels)
    
    # Separate CPU vs CoreML data
    cpu_data = df_filtered[df_filtered['provider_mode'] == 'cpu_only'].sort_values('model')
    coreml_data = df_filtered[df_filtered['provider_mode'] == 'coreml_plus_cpu'].sort_values('model')
    
    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = cpu_data['model_label'].values
    x = np.arange(len(models))
    width = 0.35
    
    # Colors
    cpu_color = '#3498db'
    coreml_color = '#e74c3c'
    
    # ============ SUBPLOT 1: Latency (lower is better) ============
    bars1_cpu = ax1.bar(x - width/2, cpu_data['latency_p50_ms'], width, 
                        label='CPU', color=cpu_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars1_coreml = ax1.bar(x + width/2, coreml_data['latency_p50_ms'], width,
                           label='CoreML+CPU', color=coreml_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Latency p50 (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Inference Latency\n(Lower is Better)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1_cpu, bars1_coreml]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ============ SUBPLOT 2: Throughput (higher is better) ============
    bars2_cpu = ax2.bar(x - width/2, cpu_data['throughput_samples_per_s'], width,
                        label='CPU', color=cpu_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2_coreml = ax2.bar(x + width/2, coreml_data['throughput_samples_per_s'], width,
                           label='CoreML+CPU', color=coreml_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Throughput (samples/sec)', fontsize=12, fontweight='bold')
    ax2.set_title('Inference Throughput\n(Higher is Better)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars2_cpu, bars2_coreml]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Overall title
    fig.suptitle('Transformer Models: CPU vs CoreML Performance on Apple M2\n(batch=1, seq_len=128)', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path('results/plots/m2_coreml_vs_cpu_consolidated.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Consolidated plot saved to {output_path}")
    
    plt.close()
    
    return output_path


def main():
    print("Loading CSV files...")
    df = load_all_csvs()
    
    print(f"\nTotal records: {len(df)}")
    print(f"Models: {df['model'].unique()}")
    print(f"Provider modes: {df['provider_mode'].unique()}")
    
    print("\nGenerating consolidated plot...")
    create_consolidated_plot(df)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
