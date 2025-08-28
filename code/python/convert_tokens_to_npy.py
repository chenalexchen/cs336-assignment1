#!/usr/bin/env python3
"""
Convert tokenized text files to NumPy arrays for efficient training.
"""

import numpy as np
import os
from pathlib import Path

def convert_tokens_to_npy(input_file: str, output_file: str):
    """Convert token ID file to NumPy array."""
    print(f"Converting {input_file} -> {output_file}")
    
    # Read token IDs
    with open(input_file, 'r') as f:
        tokens = [int(x) for x in f.read().split()]
    
    print(f"  Loaded {len(tokens):,} tokens")
    
    # Convert to NumPy array with appropriate dtype
    tokens_array = np.array(tokens, dtype=np.uint32)
    
    # Save as .npy file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, tokens_array)
    
    print(f"  Saved to {output_file} ({tokens_array.nbytes:,} bytes)")
    return len(tokens)

def main():
    """Convert both training and validation datasets."""
    
    # Define paths
    base_path = "training_data/tiny_stories_10000"
    
    train_ids = f"{base_path}/train/tokens.ids"
    train_npy = f"{base_path}/train/tokens.npy"
    
    val_ids = f"{base_path}/validation/tokens.ids"
    val_npy = f"{base_path}/validation/tokens.npy"
    
    print("🔄 Converting tokenized datasets to NumPy format")
    print("=" * 50)
    
    total_tokens = 0
    
    # Convert validation dataset (already completed)
    if os.path.exists(val_ids):
        val_tokens = convert_tokens_to_npy(val_ids, val_npy)
        total_tokens += val_tokens
    else:
        print(f"⚠️  Validation file not found: {val_ids}")
    
    # Convert training dataset (when ready)
    if os.path.exists(train_ids):
        train_tokens = convert_tokens_to_npy(train_ids, train_npy)
        total_tokens += train_tokens
    else:
        print(f"⚠️  Training file not found: {train_ids}")
        print("   Waiting for tokenization to complete...")
    
    print("=" * 50)
    print(f"✅ Conversion complete! Total tokens: {total_tokens:,}")
    
    if os.path.exists(train_npy) and os.path.exists(val_npy):
        print(f"📁 Ready for training:")
        print(f"   Training data: {train_npy}")
        print(f"   Validation data: {val_npy}")

if __name__ == "__main__":
    main()