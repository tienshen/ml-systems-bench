# ML Systems Bench

A comprehensive benchmarking framework for evaluating Large Language Model (LLM) performance on CPU and GPU hardware.

## Features

- 🚀 Benchmark ONNX models on CPU and CUDA devices
- 📊 Collect detailed performance metrics (latency, throughput, memory usage)
- 🔄 Easy model export from HuggingFace to ONNX format
- 📈 Extensible architecture for custom backends and metrics
- 💾 Save and analyze benchmark results

## Project Structure

```
ml-systems-bench/
├── README.md
├── requirements.txt
├── scripts/
│   ├── export_to_onnx.py      # Export HuggingFace models to ONNX
│   └── run_benchmarks.py      # Run benchmarks on models
├── bench/
│   ├── __init__.py
│   ├── backends/              # Backend implementations
│   │   ├── __init__.py
│   │   ├── base_runner.py     # Abstract base class
│   │   ├── cpu_runner.py      # CPU inference runner
│   │   └── cuda_runner.py     # CUDA inference runner
│   ├── models/                # Model loading utilities
│   │   ├── __init__.py
│   │   ├── model_loader.py    # HuggingFace model loader
│   │   └── onnx_paths.py      # ONNX path management
│   ├── metrics.py             # Metrics collection
│   └── plotting.py            # Visualization (TBD)
├── models/                    # Store ONNX models here
├── results/
│   ├── raw/                   # Raw benchmark results (JSON)
│   └── plots/                 # Generated plots
└── notebooks/                 # Jupyter notebooks for analysis
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ml-systems-bench
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Optional: CUDA Support

For GPU benchmarking, install ONNX Runtime with CUDA support:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

Note: Requires CUDA toolkit to be installed on your system.

## Quick Start

### 1. Export a Model to ONNX

Export a HuggingFace model to ONNX format:

```bash
python scripts/export_to_onnx.py bert-base-uncased
```

Options:
- `--output-name`: Custom output filename
- `--max-length`: Maximum sequence length (default: 128)
- `--opset-version`: ONNX opset version (default: 14)
- `--cache-dir`: HuggingFace cache directory

Example with options:
```bash
python scripts/export_to_onnx.py distilbert-base-uncased \
    --output-name distilbert.onnx \
    --max-length 256
```

### 2. Run Benchmarks

Run benchmarks on an exported model:

```bash
python scripts/run_benchmarks.py --model bert-base-uncased
```

Options:
- `--device`: Device to use (`cpu` or `cuda`)
- `--num-iterations`: Number of benchmark iterations (default: 100)
- `--warmup-iterations`: Number of warmup iterations (default: 5)
- `--batch-size`: Input batch size (default: 1)
- `--seq-length`: Input sequence length (default: 128)
- `--num-threads`: Number of CPU threads (default: auto)
- `--output`: Output file for results

Example:
```bash
python scripts/run_benchmarks.py \
    --model bert-base-uncased \
    --device cpu \
    --num-iterations 100 \
    --batch-size 1 \
    --seq-length 128 \
    --num-threads 4
```

### 3. List Available Models

```bash
python scripts/run_benchmarks.py --list-models
```

## Usage Examples

### Benchmark CPU vs CUDA

```bash
# Benchmark on CPU
python scripts/run_benchmarks.py --model bert-base-uncased --device cpu

# Benchmark on CUDA (requires onnxruntime-gpu)
python scripts/run_benchmarks.py --model bert-base-uncased --device cuda
```

### Custom Batch Sizes and Sequence Lengths

```bash
python scripts/run_benchmarks.py \
    --model distilbert-base-uncased \
    --batch-size 8 \
    --seq-length 256
```

### Different Thread Counts

```bash
# Single thread
python scripts/run_benchmarks.py --model bert-base-uncased --num-threads 1

# Multiple threads
python scripts/run_benchmarks.py --model bert-base-uncased --num-threads 8
```

## Benchmark Metrics

The framework collects the following metrics:

- **Latency**: Per-sample inference time (mean, min, max)
- **Throughput**: Samples processed per second
- **Memory Usage**: Memory consumption during inference
- **Device Information**: Device type and configuration

Results are saved as JSON files in `results/raw/` with timestamps.

## Extending the Framework

### Adding a New Backend

1. Create a new file in `bench/backends/` (e.g., `openvino_runner.py`)
2. Inherit from `BaseRunner`
3. Implement `load_model()` and `run_inference()` methods

Example:
```python
from .base_runner import BaseRunner

class OpenVINORunner(BaseRunner):
    def __init__(self, model_path: str):
        super().__init__(model_path, "openvino")
    
    def load_model(self):
        # Implementation
        pass
    
    def run_inference(self, input_data):
        # Implementation
        pass
```

### Adding Custom Metrics

Extend the `BenchmarkMetrics` class in `bench/metrics.py` to add custom metrics collection and analysis.

## Roadmap

- [ ] Add visualization utilities in `plotting.py`
- [ ] Support for more backend types (OpenVINO, TensorRT)
- [ ] Multi-GPU benchmarking
- [ ] Automated comparison reports
- [ ] Power consumption measurements
- [ ] Integration with MLflow for experiment tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ml_systems_bench,
  title = {ML Systems Bench: LLM Performance Benchmarking Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/ml-systems-bench}
}
```

## Troubleshooting

### ONNX Export Issues

If you encounter errors during export:
- Ensure the model is supported by ONNX
- Try a different opset version
- Check PyTorch and ONNX versions are compatible

### CUDA Not Available

If CUDA benchmarks fail:
- Verify CUDA toolkit is installed
- Install `onnxruntime-gpu` instead of `onnxruntime`
- Check GPU drivers are up to date

### Memory Issues

For large models:
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing (if applicable)

## Contact

For questions or issues, please open an issue on GitHub.
