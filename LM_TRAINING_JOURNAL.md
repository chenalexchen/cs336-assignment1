# Language Model Training Journal

## Project Overview
This journal documents the complete process of training my first transformer language model on TinyStories dataset using CS336 Assignment 1 codebase.

**Hardware:** GTX 1080 Ti (8GB VRAM)  
**Dataset:** TinyStories (GPT-4 generated stories)  
**Tokenizer:** BPE with 10,000 vocabulary size  

---

## Phase 1: BPE Tokenizer Training

### Step 1: Train BPE Tokenizer
**Date:** 2025-08-28  
**Duration:** 349.2 seconds (~6 minutes)

```bash
# Create output directory
mkdir -p tokenizer_output/tiny_stories_10000/training

# Train BPE tokenizer with vocab size 10,000
uv run python code/python/train_bpe_tokenizer.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output tokenizer_output/tiny_stories_10000/training \
  --vocab-size 10000
```

**Results:**
- ✅ Final vocabulary size: 10,000
- ✅ Number of merges learned: 9,743
- ✅ Token breakdown: 256 base bytes + 1 special token + 9,743 learned merges
- ✅ Files created:
  - `tokenizer_output/tiny_stories_10000/training/vocab.json`
  - `tokenizer_output/tiny_stories_10000/training/merges.txt`
  - `tokenizer_output/tiny_stories_10000/training/training_stats.txt`

**Test Results:**
- Input: "Hello world! This is a test of the BPE tokenizer."
- Output: 17 tokens with lossless encoding/decoding ✅

---

## Phase 2: Dataset Tokenization

### Step 2: Tokenize Training and Validation Datasets
**Date:** 2025-08-28

```bash
# Create directory structure
mkdir -p training_data/tiny_stories_10000/train training_data/tiny_stories_10000/validation

# Tokenize validation dataset (completed in ~7 minutes)
uv run python code/python/bpe_tokenize.py \
  --vocab tokenizer_output/tiny_stories_10000/training/vocab.json \
  --merges tokenizer_output/tiny_stories_10000/training/merges.txt \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output training_data/tiny_stories_10000/validation/tokens.ids \
  --output-format ids --no-stats

# Tokenize training dataset (still in progress...)
uv run python code/python/bpe_tokenize.py \
  --vocab tokenizer_output/tiny_stories_10000/training/vocab.json \
  --merges tokenizer_output/tiny_stories_10000/training/merges.txt \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output training_data/tiny_stories_10000/train/tokens.ids \
  --output-format ids --no-stats
```

**Validation Dataset Results:**
- ✅ Tokenization completed: 12,255,487 tokens
- ✅ Output file: `training_data/tiny_stories_10000/validation/tokens.ids`

**Training Dataset Status:**
- ⏳ Still processing (large 2.1GB file)
- 📍 Running in background (bash_2)

### Step 3: Convert to NumPy Format
**Date:** 2025-08-28

Created conversion script and moved it to proper location:
```bash
# Move script to code directory
mv convert_tokens_to_npy.py code/python/convert_tokens_to_npy.py

# Convert validation data (completed)
uv run python code/python/convert_tokens_to_npy.py
```

**Validation Conversion Results:**
- ✅ Loaded 12,255,487 tokens
- ✅ Saved to `training_data/tiny_stories_10000/validation/tokens.npy` (49,021,948 bytes)
- ⏳ Training data awaiting tokenization completion

---

## Phase 3: Training Setup

### Step 4: Prepare Training Infrastructure
**Date:** 2025-08-28

```bash
# Create checkpoint directory
mkdir -p checkpoints/my_first_model
```

### Model Parameters (GTX 1080 Ti Optimized)
**Rationale:** Conservative settings for 8GB GPU memory, designed for first-time training

```bash
# Training command (ready to execute)
uv run python code/python/train.py \
  --d_model 256 \           # Small model dimension for memory efficiency
  --num_heads 8 \           # 8 attention heads
  --d_ff 1024 \            # Feed-forward dimension
  --num_layers 6 \          # 6 transformer layers
  --vocab_size 10000 \      # Matches our BPE tokenizer
  --context_length 256 \    # Short sequences for memory
  --batch_size 16 \         # Conservative batch size
  --learning_rate 3e-4 \    # Standard transformer learning rate
  --weight_decay 0.1 \      # AdamW weight decay
  --warmup_steps 500 \      # Warmup for 500 steps
  --max_steps 10000 \       # Total training steps
  --train_data training_data/tiny_stories_10000/train/tokens.npy \
  --val_data training_data/tiny_stories_10000/validation/tokens.npy \
  --checkpoint_dir checkpoints/my_first_model \
  --log_interval 50 \       # Log every 50 steps
  --eval_interval 500 \     # Evaluate every 500 steps
  --dtype float16 \         # Half precision for memory savings
  --device cuda
```

**Expected Model Stats:**
- Parameters: ~25M parameters
- Memory usage: ~4-5GB with float16
- Training time: ~2-3 hours for 10k steps

---

## Current Status
**Date:** 2025-08-28, 04:45 AM

✅ **Completed:**
- BPE tokenizer training (10K vocab)
- Validation dataset tokenization (12.2M tokens)
- Data conversion script creation and organization
- NumPy conversion for validation data
- Training infrastructure setup
- Model parameters optimization for GTX 1080 Ti

⏳ **In Progress:**
- Training dataset tokenization (background process)

🚀 **Next Steps:**
1. Wait for training tokenization completion
2. Convert training tokens to NumPy format
3. Launch model training
4. Monitor training progress and loss curves
5. Save checkpoints every 500 steps
6. Test text generation after training

---

## Issues & Notes

### GPU Compatibility Notes
- GTX 1080 Ti does NOT support bfloat16 natively
- Using float16 instead for memory efficiency
- Hardware supports compute capability 6.1 (Pascal architecture)
- bfloat16 would require software emulation (much slower)

### File Organization
- All tokenizer outputs: `tokenizer_output/tiny_stories_10000/training/`
- All training data: `training_data/tiny_stories_10000/{train,validation}/`
- All checkpoints: `checkpoints/my_first_model/`
- All scripts: `code/python/`

### Commands Reference
```bash
# Quick data conversion check
uv run python code/python/convert_tokens_to_npy.py

# Monitor training (when started)
tail -f checkpoints/my_first_model/training.log

# Test tokenizer
uv run python code/python/bpe_tokenize.py \
  --vocab tokenizer_output/tiny_stories_10000/training/vocab.json \
  --merges tokenizer_output/tiny_stories_10000/training/merges.txt \
  --interactive
```