#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting DINO-Guided Face Generator PoC..."

echo ""
echo "--- Milestone 1: DINOv2 Feature Pyramid Pre-computation ---"

# Activate virtual environment and run the preprocessing script
source .venv/bin/activate
python scripts/preprocess_pyramids.py

# Future milestones will be called here. For example:
# echo ""
# echo "--- Milestone 2: Foundational Model Training ---"
# python scripts/train_model.py

echo ""
echo "PoC execution finished."
