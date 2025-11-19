#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

ENV_NAME="kedro_project"

# Check if environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=3.11
fi

# Activate the environment
echo "Activating Conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping dependency installation."
fi

# Run Kedro
echo "Running Kedro pipeline..."
kedro run
