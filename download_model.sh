#!/usr/bin/env bash
# Thin wrapper — real logic lives in scripts/download_model.py (cross-platform:
# Linux/macOS/Git-Bash/WSL AND native Windows PowerShell all share it).
#
# Usage:
#   ./download_model.sh                     # default: models/bge-large-en-v1.5
#   HF_ENDPOINT=https://hf-mirror.com ./download_model.sh   # 国内镜像
set -euo pipefail
cd "$(dirname "$0")"
exec python3 scripts/download_model.py "$@"
