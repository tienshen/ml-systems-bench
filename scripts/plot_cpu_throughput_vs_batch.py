import os
import json
import matplotlib.pyplot as plt

MODEL_NAME = "bert-base-uncased"
SEQ_LEN = 128
batches = [1, 8]

def load_summary(batch_size):
    path = os.path.join(
        "results",
        "raw",
        f"{MODEL_NAME}_cpu_bs{batch_size}_seq{SEQ_LEN}.json",
    )
    with open(path, "r") as f:
        return json.load(f)

def main():
    throughputs = []
    for bs in batches:
        summary = load_summary(bs)
        throughputs.append(summary["throughput"])

    plt.figure()
    plt.bar([str(b) for b in batches], throughputs)
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (inferences/sec)")
    plt.title(f"CPU throughput vs batch size ({MODEL_NAME})")

    os.makedirs("results/plots", exist_ok=True)
    out_path = os.path.join("results", "plots", f"{MODEL_NAME}_cpu_throughput_bs.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
