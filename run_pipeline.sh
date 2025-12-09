#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# 1. SupCon Embedder Training (Stage 1)
echo "=== Step 1: Starting SupCon Embedder Training (Stage 1) ==="
uv run src/trainModel.py --mode supcon --epochs 128
echo "Stage 1 Complete. Embedder saved to stance_embedder.pth"

# 2. Classifier Training (Stage 2)
echo "=== Step 2: Starting Classifier Training (Stage 2) ==="
# Note: Stage 2 loads from stance_embedder.pth as configured in trainModel.py default or we explicitly set it if needed.
# Based on current trainModel.py default, --checkpoint default is stance_embedder.pth.
uv run src/trainModel.py --mode classifier --checkpoint stance_embedder.pth --epochs 32
echo "Stage 2 Complete. Classifier saved to stance_classifier.pth"

# 3. Inference and Vectorization
echo "=== Step 3: Inference and Vectorization ==="
uv run src/run_inference.py
echo "Pipeline Complete. Results saved to embeddings_with_predictions.parquet"
