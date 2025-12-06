"""Script to export HuggingFace models to ONNX format."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.models import ModelLoader, ensure_models_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace models to ONNX format"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="HuggingFace model name (e.g., 'bert-base-uncased', 'distilbert-base-uncased')"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: model_name.onnx)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache HuggingFace models"
    )
    
    args = parser.parse_args()
    
    # Determine output name
    if args.output_name:
        output_name = args.output_name
    else:
        # Use model name, replace / with _
        output_name = args.model_name.replace("/", "_")
    
    if not output_name.endswith(".onnx"):
        output_name += ".onnx"
    
    # Ensure models directory exists
    models_dir = ensure_models_dir()
    output_path = models_dir / output_name
    
    print(f"Exporting {args.model_name} to ONNX...")
    print(f"Output path: {output_path}")
    
    try:
        # Load model
        loader = ModelLoader(args.model_name, cache_dir=args.cache_dir)
        loader.load_from_huggingface()
        
        # Create sample input
        sample_input = loader.create_sample_input(max_length=args.max_length)
        
        # Export to ONNX
        loader.export_to_onnx(
            str(output_path),
            input_sample=sample_input,
            opset_version=args.opset_version
        )
        
        print(f"\n✓ Successfully exported to {output_path}")
        print(f"Model size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Error exporting model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
