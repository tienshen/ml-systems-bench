"""Utilities for loading and preparing models."""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel


class ModelLoader:
    """Helper class for loading and exporting models."""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            model_name: HuggingFace model name or path
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        
    def load_from_huggingface(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        print(f"Loading {self.model_name} from HuggingFace...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        self.model.eval()
        print("Model loaded successfully")
        
    def export_to_onnx(
        self,
        output_path: str,
        input_sample: Optional[Dict[str, torch.Tensor]] = None,
        opset_version: int = 14
    ) -> None:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_sample: Sample input for tracing (None for default)
            opset_version: ONNX opset version
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_from_huggingface() first.")
        
        # Create sample input if not provided
        if input_sample is None:
            input_sample = self.tokenizer(
                "This is a sample text for export",
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True
            )
        
        # Get input names
        input_names = list(input_sample.keys())
        output_names = ["output"]
        
        # Create dynamic axes for batch and sequence
        dynamic_axes = {
            name: {0: "batch_size", 1: "sequence_length"}
            for name in input_names
        }
        dynamic_axes["output"] = {0: "batch_size", 1: "sequence_length"}
        
        print(f"Exporting to ONNX: {output_path}")
        
        # Export
        torch.onnx.export(
            self.model,
            tuple(input_sample.values()),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        print(f"Model exported successfully to {output_path}")
    
    def create_sample_input(
        self,
        text: str = "Sample input text",
        max_length: int = 128
    ) -> Dict[str, Any]:
        """
        Create sample input for the model.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_from_huggingface() first.")
        
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
