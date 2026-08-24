#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Memory Skill — One-Command Environment Setup
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./setup.sh              # Create venv + install deps + link model
#   ./setup.sh --no-model   # Skip model setup (embedder uses SHA-256 fallback)
#   ./setup.sh --model-path /path/to/model  # Use custom model path
#
# What this does:
#   1. Creates a Python 3.11+ virtual environment at .venv/
#   2. Installs all dependencies from requirements.txt
#   3. Sets up the ONNX embedding model (bge-large-en-v1.5)
#   4. Copies .env.example → .env if .env doesn't exist
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Flags ────────────────────────────────────────────────────────────────────
SKIP_MODEL=false
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-model) SKIP_MODEL=true; shift ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./setup.sh [--no-model] [--model-path PATH]"
            echo ""
            echo "Options:"
            echo "  --no-model        Skip embedding model setup (SHA-256 fallback)"
            echo "  --model-path PATH Use a custom model directory or file"
            exit 0
            ;;
        *) error "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Step 1: Python version check ─────────────────────────────────────────────
info "Checking Python version..."
PYTHON=$(command -v python3.11 || command -v python3 || command -v python)
PYTHON_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo "0")

echo "  Using: $("$PYTHON" --version)"

if [ "$PYTHON_MINOR" -lt 10 ]; then
    error "Python 3.10+ required. Found: $("$PYTHON" --version)"
    error "Install Python 3.11+: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# ── Step 2: Virtual environment ──────────────────────────────────────────────
if [ -d ".venv" ]; then
    info "Virtual environment already exists at .venv/"
    read -p "  Recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        info "Creating virtual environment..."
        "$PYTHON" -m venv .venv
    fi
else
    info "Creating virtual environment at .venv/"
    "$PYTHON" -m venv .venv
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# ── Step 3: Upgrade pip ─────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip -q

# ── Step 4: Install dependencies ─────────────────────────────────────────────
info "Installing dependencies from requirements.txt..."
pip install -r requirements.txt -q

info "Installing project in editable mode..."
pip install -e . -q

# ── Step 5: Optional ONNX dependencies ───────────────────────────────────────
info "Installing ONNX + tokenizer (optional)..."
pip install -e ".[onnx]" -q 2>/dev/null || warn "onnx extras not available (ONNX embedding requires model)"

# ── Step 6: Model setup ──────────────────────────────────────────────────────
if [ "$SKIP_MODEL" = true ]; then
    warn "运行模式: 无嵌入模型（SHA-256 fallback，语义检索/去重/学习判定失效，仅 BM25 可用）"
elif [ -n "$MODEL_PATH" ]; then
    info "Using custom model path: $MODEL_PATH"
    export MEMORY_MODEL_PATH="$MODEL_PATH"
else
    MODEL_DIR="models/bge-large-en-v1.5"

    if [ -d "$MODEL_DIR" ] || [ -L "$MODEL_DIR" ]; then
        info "Model already present at $MODEL_DIR"
    elif [ -d "/home/baaai/models/bge-large-en-v1.5" ]; then
        info "Found model at /home/baaai/models/bge-large-en-v1.5 → symlinking..."
        mkdir -p models
        ln -sfn /home/baaai/models/bge-large-en-v1.5 "$MODEL_DIR"
    else
        info "未找到本地模型，尝试自动下载 bge-large-en-v1.5（约 1.3GB）…"
        # 直连 HuggingFace，失败则自动换 hf-mirror.com 镜像
        if ! bash download_model.sh; then
            warn "直连 HuggingFace 失败，改用 hf-mirror.com 镜像重试…"
            if ! HF_ENDPOINT=https://hf-mirror.com bash download_model.sh; then
                error "模型下载失败。手动修复："
                error "  bash download_model.sh                     # 直连"
                error "  HF_ENDPOINT=https://hf-mirror.com bash download_model.sh   # 国内镜像"
                error "  或显式接受降级模式：./setup.sh --no-model"
                exit 1
            fi
        fi

        # 下载完成后校验 ONNX 文件（转换产物可能位于 onnx/ 子目录或模型根目录）
        if [ ! -f "$MODEL_DIR/onnx/model.onnx" ] && [ ! -f "$MODEL_DIR/model.onnx" ]; then
            error "模型下载完成但未找到 ONNX 文件（检查 $MODEL_DIR/onnx/model.onnx 或 $MODEL_DIR/model.onnx）"
            error "请确认 optimum 转换成功，或手动重试：bash download_model.sh"
            exit 1
        fi
        info "模型已就绪：$MODEL_DIR"
    fi
fi

# ── Step 7: .env setup ───────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    info "Creating .env from .env.example..."
    cp .env.example .env
    warn "Edit .env to set IMPORTANCE_API_KEY for LLM importance gate."
else
    info ".env already exists — skipping."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Memory Skill environment is ready!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Activate:  source .venv/bin/activate"
echo "  Status:    memory status"
echo "  Search:    memory search \"your query\""
echo "  Ingest:    memory ingest \"hello world\""
echo "  Weave:     memory weave \"current context\""
echo ""
