#!/bin/bash
# This is the definitive setup script that installs a consistent "time capsule"
# of packages from mid-2023 known to be compatible with each other.
set -e

# --- 1. Install Miniconda ---
echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
conda init bash
source ~/.bashrc

# --- 2. Accept Terms of Service ---
echo "Accepting Anaconda Terms of Service..."
conda config --set auto_update_conda false
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

# --- 3. Create Conda Environment ---
ENV_NAME="ironic_suppression"
conda create -n $ENV_NAME python=3.10 -y
source $HOME/miniconda3/bin/activate $ENV_NAME

# --- 4. Install the Time-Locked Package Set ---
echo "Installing the complete, compatible package set..."
pip install \
  torch==2.0.1 \
  torchvision==0.15.2

pip install \
  transformer-lens==1.4.0 \
  transformers==4.30.2 \
  accelerate==0.21.0 \
  safetensors==0.3.1 \
  pandas==2.0.3 \
  numpy==1.24.3 \
  scipy==1.10.1 \
  openai==1.3.5 \
  httpx==0.24.1 \
  nltk==3.8.1 \
  tqdm==4.66.1 \
  einops \
  matplotlib \
  seaborn

# --- 5. Download NLTK Data ---
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

echo "Definitive setup complete. Environment '$ENV_NAME' is ready."