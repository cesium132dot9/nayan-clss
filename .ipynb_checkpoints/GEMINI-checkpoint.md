## GEMINI.md

### Project Overview

YOU ARE A LEAD RESEARCH SCIENTIST AT ANTHROPIC, OPENAI, DEEPMIND, ETC.

This project is a machine learning research experiment focused on the "ironic rebound" effect in large language models. The goal is to analyze whether instructing a model *not* to mention a specific concept makes it more likely to do so. The research compares the log probabilities of forbidden concepts in negative vs. neutral/positive prompts across various transformer architectures.

The core technologies used are Python, PyTorch, and the Hugging Face `transformers` and `transformer_lens` libraries. The project uses a dataset named `negation_dataset.csv`.

### Building and Running

**1. Environment Setup:**

To set up the environment, install the required Python packages:

```bash
pip install pandas numpy torch matplotlib datasets transformer_lens scipy statsmodels scikit-learn accelerate einops bitsandbytes transformers tokenizers safetensors huggingface_hub
```

You will also need to set your Hugging Face token as an environment variable to access certain models:

```bash
export HF_TOKEN=<your_hugging_face_token>
```

**2. Running Experiments:**

The main experiment can be run using the `script.py` file:

```bash
python script.py
```

This script will iterate through a list of models, run two main experiments (E1: mention-controlled contrast, and E2: load/distractor effects), and save the results in separate directories for each model (e.g., `results_gpt2-small_simplified/`).

Alternatively, you can run the experiments from the `Nayan-code` subdirectory, which uses a different set of models:

```bash
cd Nayan-code
python script.py
```

**3. Interactive Analysis:**

The `main.ipynb` Jupyter notebook can be used for interactive analysis and visualization of the results.

### Development Conventions

*   **Code Structure:** The core experiment logic is encapsulated in the `SimplifiedIronicReboundExperiments` class in `script.py`. This class handles model loading, experiment execution, statistical analysis, and visualization.
*   **Model Configuration:** Models are configured in the `run_simplified_pipeline` method within `script.py`.
*   **Results:** Experiment results, including CSV files with raw data and PNG files with visualizations, are saved in directories named `results_{model_name}_simplified/`.
*   **Statistical Analysis:** The project uses bootstrap confidence intervals and permutation testing for statistical significance.
*   **Error Handling:** The code includes robust error handling, especially for CUDA out-of-memory issues during model loading.
