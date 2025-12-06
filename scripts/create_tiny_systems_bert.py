#!/usr/bin/env python3
"""
Create a tiny BERT-style model + tokenizer for benchmarking / systems experiments.

- vocab_size = 8000  (well under CoreML's 16384 limit)
- hidden_size = 128
- num_hidden_layers = 2
- intermediate_size = 4 * hidden_size
- num_attention_heads = 4  (128 / 4 = 32 per head)

Outputs a Hugging Face-compatible directory:

  <out_dir>/
    config.json
    pytorch_model.bin
    vocab.txt
    tokenizer_config.json
    special_tokens_map.json
    tokenizer.json

Usage (from repo root):

  python3 scripts/create_tiny_systems_bert.py \
    --out-dir tiny-systems-bert
"""

import argparse
from pathlib import Path

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast


def build_vocab_tokens(vocab_size: int):
    """
    Build a simple BERT-like WordPiece vocab:

    - [PAD], [UNK], [CLS], [SEP], [MASK]
    - [unused0] .. [unused9]
    - tok0000, tok0001, ...
    """
    if vocab_size < 16:
        raise ValueError("vocab_size should be at least 16 for this template.")

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    unused_tokens = [f"[unused{i}]" for i in range(10)]  # [unused0]..[unused9]

    base = special_tokens + unused_tokens
    num_remaining = vocab_size - len(base)

    other_tokens = [f"tok{i:04d}" for i in range(num_remaining)]
    return base + other_tokens


def write_vocab_file(vocab_tokens, vocab_path: Path):
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        for tok in vocab_tokens:
            f.write(tok + "\n")


def create_tokenizer(out_dir: Path, vocab_size: int = 8000):
    vocab_tokens = build_vocab_tokens(vocab_size)
    vocab_path = out_dir / "vocab.txt"
    write_vocab_file(vocab_tokens, vocab_path)

    # Create a fast tokenizer from the vocab file
    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_path),
        do_lower_case=True,
    )

    # Set special tokens explicitly
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
    )

    tokenizer.save_pretrained(out_dir)
    print(f"[INFO] Saved tokenizer to {out_dir}")


def create_model(out_dir: Path, vocab_size: int = 8000):
    """
    Create a randomly initialized tiny BERT encoder suitable for benchmarking.
    """
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,  # 4 * hidden_size
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
    )

    model = BertModel(config)
    # Random weights are fine for benchmarking; no training needed
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"[INFO] Saved model to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="tiny-systems-bert",
        help="Output directory for the HF-style model folder.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size for the tiny model (default: 8000).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    vocab_size = args.vocab_size

    print(f"[INFO] Creating tiny BERT-style model in {out_dir}")
    print(f"[INFO] vocab_size={vocab_size}, hidden_size=128, layers=2, heads=4")

    create_model(out_dir, vocab_size=vocab_size)
    create_tokenizer(out_dir, vocab_size=vocab_size)

    # Quick sanity check: load back with AutoModel/AutoTokenizer
    from transformers import AutoModel, AutoTokenizer

    _ = AutoModel.from_pretrained(out_dir)
    _ = AutoTokenizer.from_pretrained(out_dir)

    print("[INFO] Sanity load passed (AutoModel + AutoTokenizer).")
    print("[DONE] Tiny systems BERT created successfully.")


if __name__ == "__main__":
    main()
