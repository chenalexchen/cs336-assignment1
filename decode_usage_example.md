# decode.py Usage Guide

The `decode.py` script provides a complete CLI interface for text generation using your trained transformer language models. It can automatically load checkpoints saved by `train.py` and generate text with various sampling strategies.

## Basic Usage

### Single Text Generation
```bash
# Basic generation with minimal arguments
python code/decode.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --prompt "Once upon a time" \
    --max-tokens 50

# With custom sampling parameters  
python code/decode.py \
    --checkpoint checkpoints/checkpoint_step_10000.pt \
    --prompt "The future of AI is" \
    --max-tokens 100 \
    --temperature 0.8 \
    --top-p 0.9 \
    --device cuda
```

### Interactive Mode
```bash
# Start interactive session for multiple prompts
python code/decode.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --interactive \
    --temperature 0.7 \
    --max-tokens 75
```

### Reproducible Generation
```bash
# Set seed for consistent results
python code/decode.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --prompt "Hello world" \
    --seed 42 \
    --temperature 1.0
```

## Auto-Detection Features

### Automatic Configuration Loading
The script automatically searches for `config.json` in these locations:
1. Same directory as the checkpoint file
2. Parent directory of the checkpoint file
3. Specified via `--config` argument

### Automatic Tokenizer Detection
The script searches for `vocab.json` and `merges.txt` in:
1. Same directory as the checkpoint
2. Parent directory of the checkpoint  
3. `tokenizer_output/` directory
4. Current directory
5. Specified via `--vocab` and `--merges` arguments

## Command-Line Arguments

### Required Arguments
- `--checkpoint`: Path to model checkpoint (.pt file)

### Generation Parameters
- `--prompt`: Input text prompt (required unless using `--interactive`)
- `--max-tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature - higher = more creative (default: 1.0)
- `--top-p`: Top-p (nucleus) sampling threshold (optional)
- `--seed`: Random seed for reproducible generation (optional)

### File Paths (Auto-detected if not specified)
- `--config`: Path to config.json file
- `--vocab`: Path to vocab.json file
- `--merges`: Path to merges.txt file

### Other Options  
- `--device`: Device to use - "cpu", "cuda", etc. (auto-detected)
- `--special-tokens`: Special tokens for tokenizer (default: ["<|endoftext|>"])
- `--interactive`: Interactive mode for multiple prompts
- `--help`: Show help message

## Examples

### Story Generation
```bash
python code/decode.py \
    --checkpoint checkpoints/tinystories_model.pt \
    --prompt "Once upon a time, in a magical forest," \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-p 0.95
```

### Code Generation
```bash
python code/decode.py \
    --checkpoint checkpoints/code_model.pt \
    --prompt "def fibonacci(n):" \
    --max-tokens 150 \
    --temperature 0.2
```

### Creative Writing
```bash
python code/decode.py \
    --checkpoint checkpoints/creative_model.pt \
    --prompt "The last human on Earth" \
    --max-tokens 300 \
    --temperature 1.2 \
    --interactive
```

## Interactive Mode Commands

When using `--interactive` mode:
- Type your prompt and press Enter to generate
- Type `quit`, `exit`, or `q` to exit
- Press Ctrl+C to exit immediately

## Integration with Training Pipeline

The decode script is designed to work seamlessly with models trained using `train.py`:

1. **Train your model:**
   ```bash
   python code/train.py \
       --train_data data/train.npy \
       --val_data data/val.npy \
       --checkpoint_dir checkpoints \
       --max_steps 50000
   ```

2. **Generate text:**
   ```bash
   python code/decode.py \
       --checkpoint checkpoints/final_checkpoint.pt \
       --prompt "Your prompt here"
   ```

The script automatically loads the model configuration from `checkpoints/config.json` and detects the tokenizer files.

## Programmatic Usage

You can also import and use the decode functions in your own scripts:

```python
from code.decode import (
    load_model_from_checkpoint, 
    load_tokenizer_from_files,
    decode_with_tokenizer
)

# Load model and tokenizer
model = load_model_from_checkpoint("checkpoints/model.pt")
tokenizer = load_tokenizer_from_files("vocab.json", "merges.txt")

# Generate text
text = decode_with_tokenizer(
    model=model,
    tokenizer=tokenizer, 
    prompt="Hello world",
    max_tokens=50,
    temperature=0.8,
    top_p=0.9
)
print(text)
```

## Troubleshooting

### Common Issues

**"Config file not found"**
- Ensure `config.json` exists in the checkpoint directory
- Or specify `--config` explicitly

**"Could not find or load tokenizer"**  
- Ensure `vocab.json` and `merges.txt` exist
- Or specify `--vocab` and `--merges` explicitly

**"CUDA out of memory"**
- Use `--device cpu` to run on CPU
- Or reduce the model size/context length

**Generation seems stuck**
- The model might be generating very long sequences
- Try using `--max-tokens` to limit generation length
- Use Ctrl+C to interrupt if needed