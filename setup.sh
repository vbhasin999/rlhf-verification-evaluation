#!/bin/bash

# Exit on error
set -e

# Install libmamba solver and set it
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

# Create and activate the environment
CONDA_OVERRIDE_CUDA=11.7 conda env create --file conda_recipe.yaml
conda activate rlhf-trojan-competition
