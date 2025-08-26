# BPE Tokenizer CLI Tools Guide

Two comprehensive CLI scripts for training and using BPE (Byte Pair Encoding) tokenizers with your transformer models.

## Scripts Overview

1. **`train_bpe_tokenizer.py`** - Train BPE tokenizers from text data
2. **`bpe_tokenize.py`** - Tokenize/detokenize text using trained tokenizers

---

## 🚀 train_bpe_tokenizer.py

### Basic Usage

```bash
# Train a tokenizer with default settings
python code/train_bpe_tokenizer.py \
    --input data/corpus.txt \
    --output my_tokenizer \
    --vocab-size 32000
```

### Complete Example
```bash
# Train tokenizer for your transformer model
python code/train_bpe_tokenizer.py \
    --input data/tinystories_train.txt \
    --output tokenizer_output \
    --vocab-size 32000 \
    --special-tokens "<|endoftext|>" "<|pad|>" "<|unk|>" \
    --test-text "Once upon a time" \
    --verbose
```

### Arguments

**Required:**
- `--input, -i`: Path to training text file
- `--output, -o`: Output directory for tokenizer files  
- `--vocab-size, -v`: Target vocabulary size

**Optional:**
- `--special-tokens, -s`: Special tokens (default: `["<|endoftext|>"]`)
- `--test-text`: Sample text to test after training
- `--verbose`: Enable verbose training output

### Output Files

The script creates these files in the output directory:
- `vocab.json` - Vocabulary mapping
- `merges.txt` - Merge rules 
- `training_stats.txt` - Training statistics

### Example Output
```
🚀 Starting BPE tokenizer training
==================================================
📁 Input file: data/corpus.txt
📁 Output directory: tokenizer_output
📊 Target vocabulary size: 32,000
🏷️  Special tokens: ['<|endoftext|>']
📏 Input file size: 15,234,567 bytes (14.5 MB)

⏳ Training BPE tokenizer...
✅ Training completed in 45.2s
📈 Final vocabulary size: 32,000
🔗 Number of merges learned: 31,743

📊 Token breakdown:
   • Base bytes (0-255): 256
   • Special tokens: 1
   • Learned merge tokens: 31,743

✓ Saved vocabulary to tokenizer_output/vocab.json
✓ Saved merges to tokenizer_output/merges.txt
✓ Saved training stats to tokenizer_output/training_stats.txt
```

---

## 🔤 bpe_tokenize.py

### Basic Tokenization

```bash
# Tokenize a text file
python code/bpe_tokenize.py \
    --vocab tokenizer_output/vocab.json \
    --merges tokenizer_output/merges.txt \
    --input document.txt \
    --output document.tokens
```

### All Usage Modes

#### 1. File Tokenization
```bash
# Tokenize file to token IDs
python code/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --input corpus.txt \
    --output corpus.tokens \
    --output-format ids

# Tokenize to JSON format
python code/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --input corpus.txt \
    --output corpus.json \
    --output-format json
```

#### 2. File Detokenization  
```bash
# Convert token IDs back to text
python code/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --input document.tokens \
    --output document_reconstructed.txt \
    --mode detokenize
```

#### 3. Direct Text Processing
```bash
# Tokenize text directly
python code/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --text "Hello world! This is a test."
```

#### 4. Interactive Mode
```bash
# Interactive tokenization session
python code/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --interactive
```

### Arguments

**Required:**
- `--vocab, -v`: Path to vocab.json file
- `--merges, -m`: Path to merges.txt file

**Input/Output:**
- `--input, -i`: Input file path
- `--output, -o`: Output file path
- `--text`: Direct text input

**Modes:**
- `--mode`: `tokenize` or `detokenize` (default: `tokenize`)
- `--interactive`: Interactive mode

**Formats:**
- `--output-format`: `ids`, `json`, or `text` (default: `ids`)
- `--input-format`: `ids` or `json` (default: `ids`)

**Options:**
- `--special-tokens, -s`: Special tokens list
- `--no-stats`: Disable statistics output

### Interactive Commands

In interactive mode:
- `encode <text>` - Tokenize text
- `decode <token_ids>` - Detokenize space-separated IDs
- `stats` - Show tokenizer statistics  
- `quit`/`exit` - Exit interactive mode

---

## 🔄 Complete Workflow

### 1. Train Tokenizer
```bash
python code/train_bpe_tokenizer.py \
    --input data/tinystories_train.txt \
    --output tokenizer_tinystories \
    --vocab-size 32000 \
    --special-tokens "<|endoftext|>"
```

### 2. Tokenize Training Data
```bash
python code/bpe_tokenize.py \
    --vocab tokenizer_tinystories/vocab.json \
    --merges tokenizer_tinystories/merges.txt \
    --input data/tinystories_train.txt \
    --output data/tinystories_train.tokens
```

### 3. Convert to NumPy (for training script)
```python
import numpy as np

# Load token IDs
with open('data/tinystories_train.tokens', 'r') as f:
    token_ids = [int(x) for x in f.read().split()]

# Save as NumPy array for train.py
np.save('data/tinystories_train.npy', np.array(token_ids, dtype=np.int32))
```

### 4. Train Model
```bash
python code/train.py \
    --train_data data/tinystories_train.npy \
    --val_data data/tinystories_val.npy \
    --vocab_size 32000 \
    --checkpoint_dir checkpoints
```

### 5. Generate Text
```bash
python code/decode.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --vocab tokenizer_tinystories/vocab.json \
    --merges tokenizer_tinystories/merges.txt \
    --prompt "Once upon a time" \
    --max-tokens 100
```

---

## 📊 Performance Tips

### For Large Datasets
```bash
# Use larger vocab sizes for better compression
python code/train_bpe_tokenizer.py \
    --input large_corpus.txt \
    --output large_tokenizer \
    --vocab-size 50000 \
    --verbose
```

### For Code/Technical Text
```bash
# Add code-specific special tokens
python code/train_bpe_tokenizer.py \
    --input code_corpus.txt \
    --output code_tokenizer \
    --vocab-size 32000 \
    --special-tokens "<|endoftext|>" "<|fim_prefix|>" "<|fim_middle|>" "<|fim_suffix|>"
```

### Memory Efficiency
The tokenizer automatically uses parallel processing for large files and includes progress tracking.

---

## 🔧 Integration with Training Pipeline

These scripts integrate seamlessly with your transformer training workflow:

1. **Train tokenizer** → `vocab.json` + `merges.txt`
2. **Tokenize datasets** → `.tokens` files  
3. **Convert to NumPy** → `.npy` files for `train.py`
4. **Train model** → checkpoints
5. **Generate text** → `decode.py` auto-detects tokenizer files

The tokenizer files are compatible with the `decode.py` script's auto-detection system!

---

## 🎯 Use Cases

- **Story Generation**: Train on TinyStories dataset
- **Code Generation**: Train on code repositories
- **Multilingual**: Train on multilingual corpora
- **Domain-Specific**: Train on scientific papers, legal text, etc.
- **Preprocessing**: Convert raw text to model-ready token sequences

Both scripts provide comprehensive error handling, progress tracking, and detailed statistics to help you understand your tokenizer's performance.