# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOU ARE A TOP RESEARCH SCIENTIST AT ANTHROPIC, OPENAI, DEEPMIND, ETC.

This repository contains machine learning research experiments focused on studying the "ironic rebound" effect in language models:

**Ironic Rebound Experiments**: Studies the phenomenon where instructing a language model NOT to mention something may paradoxically increase its likelihood of mentioning it. The experiments use various transformer models (GPT-2, OPT, Llama, Gemma) to analyze mention rates under negative vs. neutral prompts.

## Core Commands

**Environment setup:**
```bash
# Install dependencies (see SETUP.txt for complete list)
pip install pandas numpy torch matplotlib datasets transformer_lens scipy statsmodels scikit-learn accelerate einops bitsandbytes transformers tokenizers safetensors huggingface_hub

# Set HuggingFace token for gated models
export HF_TOKEN=your_token_here
```

**Run experiments:**
```bash
# Main ironic rebound experiment (full dataset)
python script.py

# Run from Nayan-code subdirectory
cd Nayan-code && python script.py

# Jupyter notebook analysis
jupyter notebook main.ipynb
```

## Architecture and Code Structure

### Core Experiment Scripts

- **`script.py`**: Main experiment implementation using `SimplifiedIronicReboundExperiments` class
  - Log probability-based evaluation approach
  - Processes full dataset (~5000 samples)
  - Uses transformer_lens HookedTransformer for model loading
  - Supports multiple models: GPT-2, OPT, Llama, Gemma

- **`ironic_rebound_experiments.py`**: Alternative implementation using `IronicReboundExperiments` class
  - Generation-based evaluation approach
  - Two experiments: short (E1) and full-length (E2) text generation
  - Mention detection in generated text
  - Comprehensive statistical analysis

- **`test_minimal.py`**: Testing and validation script
  - Basic functionality tests (dataset loading, model operations)
  - Experiment logic validation with small data subset
  - Useful for debugging and ensuring proper setup

### Key Components

**SimplifiedIronicReboundExperiments**: Main experiment class that:
- Loads models using transformer_lens with proper device management
- Processes negative vs. neutral prompts from dataset
- Calculates log probabilities for forbidden concepts
- Handles multi-token concepts with fallback tokenization
- Generates statistical comparisons and visualizations

**IronicReboundExperiments**: Alternative implementation that:
- Uses text generation instead of log probability analysis
- Implements mention detection in generated outputs
- Supports two generation lengths for comprehensive analysis
- Provides detailed statistical reporting

### Data Processing

**Dataset Structure**: Uses `negation_dataset.csv` with columns:
- `id`: Unique identifier
- `prompt_text`: The input prompt text
- `forbidden_concept`: Concept that should/shouldn't be mentioned
- `prompt_type`: Either 'negative' (instructed not to mention) or 'neutral'

**Model Handling**: 
- Uses transformer_lens for consistent model interface
- Implements robust tokenization with fallback for multi-token concepts
- Supports GPU/CPU device management with automatic fallback
- Handles model-specific tokenization quirks

## Development Practices

### Testing
- Use `test_minimal.py` to validate basic functionality before running full experiments
- Test covers dataset loading, model operations, tokenization, and log probability calculations
- Results are saved as CSV files in `results_*` directories

### Model Requirements
- **GPU recommended**: Experiments designed for CUDA, CPU fallback available
- **Memory considerations**: Large models (Llama, Gemma) require significant GPU memory
- **HuggingFace token**: Required for gated models, set via `HF_TOKEN` environment variable

### Dataset Dependencies
- Requires `negation_dataset.csv` with proper structure (id, prompt_text, forbidden_concept, prompt_type)
- Experiments expect ~5000 samples for statistical significance

## Research Context

This codebase implements experiments to study the "ironic rebound" effect in language models - the phenomenon where instructing a model NOT to mention something may paradoxically increase its likelihood of mentioning it. 

**Two evaluation approaches**:
1. **Log probability analysis**: Measures model's propensity to output forbidden concepts
2. **Generation-based analysis**: Detects mentions in generated text outputs

**Statistical methodology**: Compares mention rates between negative prompts (instructed not to mention) vs. neutral prompts across multiple transformer architectures.