#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Download the ONNX embedding model (bge-large-en-v1.5)
# ─────────────────────────────────────────────────────────────────────────────
# The model will be saved to models/bge-large-en-v1.5/
#
# Prerequisites:
#   pip install huggingface_hub optimum[onnxruntime]
#
# Usage:
#   ./download_model.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="models/bge-large-en-v1.5"
MODEL_ID="BAAI/bge-large-en-v1.5"

echo "Downloading $MODEL_ID → $MODEL_DIR ..."
mkdir -p models

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')
print('Download complete.')
"

# Convert to ONNX if model.onnx doesn't exist
if [ ! -f "$MODEL_DIR/model.onnx" ]; then
    echo "Converting to ONNX format..."
    python3 -m optimum.onnxruntime \
        --model "$MODEL_DIR" \
        --task feature-extraction \
        "$MODEL_DIR"
    echo "ONNX conversion complete."
fi

echo ""
echo "Model ready at $MODEL_DIR"
echo "Run: ./setup.sh"
