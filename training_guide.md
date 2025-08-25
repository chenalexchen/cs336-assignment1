# Transformer Language Model Training Guide

This guide explains how to use the `code/train.py` script to train transformer language models.

## Quick Start

```bash
# Basic training run
uv run python code/train.py \
  --train_data data/train.npy \
  --val_data data/val.npy \
  --d_model 512 \
  --num_heads 8 \
  --num_layers 6 \
  --max_steps 100000

# Resume from checkpoint
uv run python code/train.py \
  --train_data data/train.npy \
  --val_data data/val.npy \
  --resume_from checkpoints/checkpoint_step_50000.pt \
  --max_steps 150000

# Training with Weights & Biases logging
uv run python code/train.py \
  --train_data data/train.npy \
  --val_data data/val.npy \
  --use_wandb \
  --wandb_project my-transformer \
  --wandb_run_name experiment-1
```

## Key Features

### Memory-Efficient Data Loading
- Uses `np.memmap` to handle datasets larger than available RAM
- Automatic detection of dataset size and efficient batch sampling
- Support for both `.npy` files and memory-mapped arrays

### Comprehensive Hyperparameter Control
All major training hyperparameters are configurable via command line:

**Model Architecture:**
- `--d_model`: Hidden dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8) 
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--num_layers`: Number of transformer layers (default: 6)
- `--vocab_size`: Vocabulary size (default: 32000)
- `--context_length`: Maximum sequence length (default: 1024)
- `--rope_theta`: RoPE theta parameter (default: 10000.0)

**Training Configuration:**
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Peak learning rate (default: 3e-4)
- `--min_learning_rate`: Minimum learning rate (default: 3e-5)
- `--warmup_steps`: Learning rate warmup steps (default: 1000)
- `--max_steps`: Maximum training steps (default: 100000)
- `--grad_clip`: Gradient clipping max norm (default: 1.0)

### Advanced Training Features

**Learning Rate Scheduling:**
- Cosine annealing with linear warmup
- Configurable warmup period and decay schedule

**Gradient Optimization:**
- AdamW optimizer with configurable betas and weight decay
- Global gradient norm clipping across all parameters
- Automatic mixed precision support (float16/float32)

**Checkpointing:**
- Automatic checkpoint saving at configurable intervals
- Resume training from any checkpoint
- Saves model state, optimizer state, and training step

### Logging and Monitoring

**Console Logging:**
- Training loss, learning rate, and throughput metrics
- Configurable logging intervals
- Validation metrics at specified intervals

**Weights & Biases Integration:**
```bash
# Enable with --use_wandb flag
--use_wandb \
--wandb_project my-project \
--wandb_run_name experiment-name
```

**Metrics Tracked:**
- Training loss and learning rate
- Validation loss and perplexity  
- Training throughput (tokens/second)
- Total parameters and elapsed time

## Example Configurations

### Small Model (Testing)
```bash
uv run python code/train.py \
  --train_data data/train.npy \
  --val_data data/val.npy \
  --d_model 128 \
  --num_heads 4 \
  --d_ff 512 \
  --num_layers 2 \
  --batch_size 16 \
  --max_steps 1000 \
  --device cpu
```

### Large Model (Production)
```bash
uv run python code/train.py \
  --train_data data/train.npy \
  --val_data data/val.npy \
  --d_model 1024 \
  --num_heads 16 \
  --d_ff 4096 \
  --num_layers 12 \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --max_steps 500000 \
  --dtype float16 \
  --compile \
  --device cuda
```

### Hyperparameter Sweep
```bash
# Different learning rates
for lr in 1e-4 2e-4 5e-4 1e-3; do
  uv run python code/train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --learning_rate $lr \
    --wandb_run_name "lr_${lr}" \
    --use_wandb
done
```

## Output Structure

The training script creates the following directory structure:

```
checkpoints/
├── config.json              # Training configuration
├── checkpoint_step_5000.pt  # Periodic checkpoints
├── checkpoint_step_10000.pt
├── ...
└── final_checkpoint.pt      # Final model state
```

## Performance Optimization

**For CPU Training:**
- Use `--dtype float32` for stability
- Smaller batch sizes (8-16) for memory efficiency

**For GPU Training:**
- Use `--dtype float16` for speed and memory savings
- Enable `--compile` for additional optimization
- Larger batch sizes (32-128) for better GPU utilization

**For Large Datasets:**
- Data is automatically memory-mapped
- No need to load entire dataset into RAM
- Efficient random sampling across the full dataset

## Troubleshooting

**Out of Memory:**
- Reduce `--batch_size`
- Use `--dtype float16`
- Reduce model size (`--d_model`, `--num_layers`)

**Slow Training:**
- Enable `--compile` for GPU
- Increase `--batch_size` if memory allows
- Check device utilization

**Convergence Issues:**
- Adjust learning rate (`--learning_rate`)
- Increase warmup steps (`--warmup_steps`)
- Check gradient clipping (`--grad_clip`)

## Next Steps

After training completes, you can:
1. Load checkpoints for inference or fine-tuning
2. Analyze training curves in Weights & Biases
3. Resume training with different hyperparameters
4. Export models for deployment

The training script provides a solid foundation for experimenting with different architectures, datasets, and training strategies.