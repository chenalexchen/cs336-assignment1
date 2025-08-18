# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is CS336 Spring 2025 Assignment 1, implementing fundamental components for building transformer language models from scratch. The assignment covers neural network building blocks, tokenization (BPE), optimization, and serialization.

## Development Environment and Commands

### Environment Setup
This project uses `uv` for dependency management. All Python commands should be run with `uv run`:

```bash
# Run any Python file
uv run <python_file_path>

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_<module>.py

# Run tests with verbose output
uv run pytest -v

# Code formatting and linting
uv run ruff check
uv run ruff format
```

### Data Download
The project requires TinyStories and OpenWebText datasets:
```bash
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
```

## Code Architecture

### Implementation Pattern
This codebase follows a test-driven development pattern where:
1. **Tests define the interface**: Test files in `tests/` specify the expected API and behavior
2. **Adapters connect implementations**: `tests/adapters.py` contains adapter functions that students must implement to connect their code to the test framework
3. **Implementation goes in student code**: Student implementations should be placed in appropriate modules

### Key Components

#### Neural Network Modules (`tests/test_model.py`, `tests/test_nn_utils.py`)
- Linear layers, embeddings, attention mechanisms
- RMSNorm, SwiGLU activation, scaled dot-product attention
- Multi-head self-attention with and without RoPE (Rotary Position Embedding)
- Complete transformer blocks and language models
- All implementations should support arbitrary batch dimensions using einops/einx

#### Tokenization (`tests/test_tokenizer.py`, `tests/test_train_bpe.py`)
- BPE (Byte Pair Encoding) tokenizer implementation in `code/bpe_tokenizer.py`
- Pre-tokenization with regex patterns following GPT-2 style
- Special token handling
- Parallel processing for large datasets using multiprocessing
- Training and inference interfaces

#### Optimization (`tests/test_optimizer.py`)
- AdamW optimizer implementation
- Gradient clipping
- Learning rate scheduling (cosine with warmup)

#### Data and Training (`tests/test_data.py`)
- Batch sampling for language modeling
- Cross-entropy loss computation
- Softmax implementation

#### Serialization (`tests/test_serialization.py`)
- Model and optimizer state checkpointing
- PyTorch state dict handling

### Test Infrastructure
- **Snapshot testing**: Tests use NumPy arrays stored in `tests/_snapshots/` for deterministic verification
- **Adapter pattern**: All test functions call adapter functions in `tests/adapters.py` which must be implemented
- **Type hints**: Extensive use of `jaxtyping` for tensor shape documentation
- **Common utilities**: `tests/common.py` contains shared test utilities

### Key Dependencies
- PyTorch ~2.6.0 (with special handling for Intel Macs)
- einops/einx for tensor operations
- jaxtyping for tensor type hints
- regex for enhanced pattern matching in tokenization
- tiktoken for tokenization reference
- pytest for testing

## Development Workflow
1. Run `uv run pytest` initially - all tests should fail with `NotImplementedError`
2. Implement the adapter functions in `tests/adapters.py` to connect your implementations
3. Create your implementations in appropriate modules (follow existing patterns)
4. Run specific test files as you implement each component
5. Use the snapshot tests to verify exact numerical correctness

## Code Style
- Line length: 120 characters (configured in pyproject.toml)
- Ruff is configured for formatting and linting
- Type hints are extensively used, especially for tensor shapes
- Follow existing patterns for module organization and naming