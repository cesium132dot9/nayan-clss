# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOU ARE A LEAD RESEARCH SCIENTIST AT DEEPMIND, ANTHROPIC, OPENAI, ETC.

This repository contains machine learning research experiments focused on studying the "ironic rebound" effect in language models - the phenomenon where instructing a language model NOT to mention something may paradoxically increase its likelihood of mentioning it.

**Research Focus**: Compare log probabilities of forbidden concepts between negative prompts (instructed not to mention) vs. neutral/positive prompts across multiple transformer architectures (GPT-2, OPT, Llama, Gemma, Bloom, Pythia, Qwen).

## Core Commands

**Environment setup:**
```bash
# Install dependencies
pip install pandas numpy torch matplotlib datasets transformer_lens scipy statsmodels scikit-learn accelerate einops bitsandbytes transformers tokenizers safetensors huggingface_hub

# Set HuggingFace token for gated models (required for Llama, Gemma)
export HF_TOKEN=your_token_here

# For GPU optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Run experiments:**
```bash
# Main ironic rebound experiment (full dataset, ~5000 samples)
python script.py

# Alternative: Run from Nayan-code subdirectory (different model selection)
cd Nayan-code && python script.py

# Jupyter notebook analysis
jupyter notebook main.ipynb
```

## Architecture and Code Structure

### Core Experiment Scripts

- **`script.py`** (root): Main experiment runner with model config for Pythia, Bloom, GPT-NeoX
  - Uses `SimplifiedIronicReboundExperiments` class
  - Model configs: pythia-410m, bloom-560m, gpt-neox-20b

- **`Nayan-code/script.py`**: Alternative runner with different model selection
  - Model configs: gpt2, opt-2.7b, gemma-7b-it, llama-3-8b-instruct, qwen3-14b
  - Same experiment framework with different target models

- **`main.ipynb`**: Jupyter notebook for interactive analysis and visualization

### Key Components

**SimplifiedIronicReboundExperiments**: Main experiment class that:
- Loads models using transformer_lens HookedTransformer with robust OOM handling
- Implements two core experiments:
  - **E1**: Mention-controlled contrast (compares log probabilities across prompt types)
  - **E2**: Load/distractor effects (tests model performance with varying context lengths)
- Handles multi-token concepts with fallback tokenization strategies
- Automatic batch sizing optimization per model architecture
- Model-specific prompt formatting (Gemma, Llama, Qwen chat templates)
- Comprehensive result validation and statistical analysis
- Automatic visualization generation (histograms, line plots with confidence intervals)

### Data Processing

**Dataset Structure**: Uses `negation_dataset.csv` (~5000 samples) with columns:
- `id`: Unique identifier (0-4999)
- `prompt_type`: 'negative' (0-1665), 'neutral' (1666-3331), 'positive' (3332-4997)
- `topic`: Context topic for the prompt
- `forbidden_concept`: Target concept to avoid/mention
- `proxy_concept`: Semantically similar alternative concept
- `prompt_text`: Formatted instruction prompt
- `is_single_token`: Boolean indicating if concept is single token

**Model Handling**: 
- Uses transformer_lens HookedTransformer for consistent interface across architectures
- Robust device management with CUDA OOM fallback strategies
- Multi-token concept handling with first-token fallback
- Model-specific chat template formatting for instruction-tuned models
- Automatic batch size optimization based on model size
- Comprehensive error handling and memory cleanup

## Results Structure

Experiments generate organized output directories:
```
results_{model_name}_simplified/
├── e1.csv              # Experiment 1 results (delta scores, log probabilities)
├── e1.png              # E1 histogram visualization
├── e2.csv              # Experiment 2 results (distractor effects)
├── e2.png              # E2 line plot with confidence intervals
└── summary_stats.csv   # Aggregate statistics (means, p-values, CIs)
```

## Model Support

**Currently Supported Architectures:**
- **Small models**: pythia-410m, bloom-560m, gpt2-small
- **Medium models**: opt-2.7b, gemma-7b-it, llama-3-8b-instruct
- **Large models**: gpt-neox-20b, qwen3-14b

**Memory Requirements:**
- Small models (≤1B params): 256 batch size
- Medium models (2-8B params): 32-128 batch size  
- Large models (>10B params): 8-16 batch size
- Automatic fallback to `device_map="auto"` for OOM scenarios

## Key Implementation Details

**Statistical Analysis**: 
- Bootstrap confidence intervals (5000 samples)
- Permutation testing for significance
- LOWESS smoothing for E2 distractor length effects
- Comprehensive result validation with missing data handling

**Visualization**: 
- E1: Delta score histograms showing ironic rebound effect magnitude
- E2: Context length vs. log probability with 95% confidence intervals
- Model-specific formatting and statistical annotations