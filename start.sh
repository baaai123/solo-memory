#!/bin/bash
# Memory Skill — 一键透明代理启动
# 用法: ./start.sh [--port 8888]
# Agent 设置 OPENAI_API_BASE=http://127.0.0.1:<port>/v1 即可

set -e
PORT=${PORT:-8888}
DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv"
PY="$VENV/bin/python"

# 1: 确保虚拟环境
if [ ! -f "$PY" ]; then
    echo "创建虚拟环境..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -r "$DIR/requirements.txt"
fi

# 2: 下载模型 (如需要)
if [ ! -f "$DIR/models/bge-large-en-v1.5/onnx/model.onnx" ]; then
    echo "下载 ONNX 模型..."
    bash "$DIR/download_model.sh"
fi

# 3: 检查密钥
if [ -z "$DEEPSEEK_API_KEY" ]; then
    if [ -f "$DIR/.env" ]; then
        source "$DIR/.env"
    fi
    if [ -z "$DEEPSEEK_API_KEY" ]; then
        echo "请设置环境变量 DEEPSEEK_API_KEY"
        exit 1
    fi
fi

# 4: 启动
echo "Memory Skill — Transparent Proxy → http://127.0.0.1:$PORT"
echo ""
echo "Agent 设置: OPENAI_API_BASE=http://127.0.0.1:$PORT/v1"
echo ""
"$PY" -m memory_skill.cli proxy --port "$PORT"
