#!/usr/bin/env python3
"""
Plot CPU-only vs CoreML+CPU results on Apple M2
using results/m2_coreml_vs_cpu.csv.

Generates:
  results/plots/m2_coreml_vs_cpu_latency.png
  results/plots/m2_coreml_vs_cpu_throughput.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / "results" / "m2_coreml_vs_cpu.csv"
    out_dir = repo_root / "results" / "plots" 
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)

    # Map provider_mode to nicer labels for the legend
    label_map = {
        "cpu_only": "CPU-only",
        "coreml_plus_cpu": "CoreML + CPU fallback",
    }
    df["mode_label"] = df["provider_mode"].map(label_map)

    # Sort for nicer plotting
    df = df.sort_values(["batch_size", "seq_len", "provider_mode"])

    # For now, we only have batch_size = 1 in this CSV.
    # If you later add batch_size=8, you can loop over batch sizes.
    batch_size = 1
    df_b1 = df[df["batch_size"] == batch_size]

    # ---- Figure 1: p50 latency vs sequence length ----
    plt.figure()
    for mode, grp in df_b1.groupby("mode_label"):
        plt.plot(
            grp["seq_len"],
            grp["latency_p50_ms"],
            marker="o",
            linestyle="-",
            label=mode,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("p50 latency (ms)")
    plt.title(f"Apple M2: p50 latency vs sequence length (batch={batch_size})")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    latency_path = out_dir / "m2_coreml_vs_cpu_latency.png"
    plt.savefig(latency_path, bbox_inches="tight", dpi=150)

    # ---- Figure 2: throughput vs sequence length ----
    plt.figure()
    for mode, grp in df_b1.groupby("mode_label"):
        plt.plot(
            grp["seq_len"],
            grp["throughput_samples_per_s"],
            marker="o",
            linestyle="-",
            label=mode,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Throughput (inferences/sec)")
    plt.title(f"Apple M2: throughput vs sequence length (batch={batch_size})")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    throughput_path = out_dir / "m2_coreml_vs_cpu_throughput.png"
    plt.savefig(throughput_path, bbox_inches="tight", dpi=150)

    print(f"Saved latency plot to     {latency_path}")
    print(f"Saved throughput plot to  {throughput_path}")


if __name__ == "__main__":
    main()
