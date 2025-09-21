#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting DINO-Guided Face Generator PoC Training Pipeline..."
echo "This script assumes your source images are already in the data/raw_images/ directory."

echo ""
echo "--- Step 1: DINOv2 Feature Pyramid Pre-computation ---"

# Activate virtual environment and run the preprocessing script
source .venv/bin/activate
python scripts/preprocess_pyramids.py

echo ""
echo "--- Step 1C: Visualizing Feature Pyramids ---"
python scripts/visualize_pyramids.py

echo ""
echo "--- Milestone 2: Foundational Model Training ---"
python scripts/train_generator.py

echo ""
echo "PoC execution finished."
