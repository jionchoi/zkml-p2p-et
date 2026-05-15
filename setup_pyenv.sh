#!/bin/bash

# Default environment name is '.venv' unless you pass a name as an argument
ENV_NAME=${1:-.venv}

echo "--- Creating virtual environment: $ENV_NAME ---"
python3 -m venv "$ENV_NAME"

echo "--- Activating environment ---"
source "$ENV_NAME/bin/activate"

echo "--- Installing ipykernel ---"
pip install --upgrade pip
pip install ipykernel

echo "--- Installing necessary libs ---"
pip install pandas matplotlib ezkl torch onnx onnxruntime

echo "--- Registering kernel with Jupyter ---"
# This makes the environment selectable in the Jupyter menu
python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "--- Setup Complete! ---"
echo "To use this environment in your terminal, run: source $ENV_NAME/bin/activate"
echo "In Jupyter, select the 'Python ($ENV_NAME)' kernel."
