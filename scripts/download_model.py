#!/usr/bin/env python3
"""Download bge-large-en-v1.5 and convert to ONNX — cross-platform.

Replaces download_model.sh for native Windows installs; download_model.sh is a
thin wrapper that delegates here so every flow (setup.sh, opencode plugin,
ccmp bootstrap, Windows PowerShell) shares ONE implementation.

Behavior (mirrors the old download_model.sh):
  1. snapshot_download BAAI/bge-large-en-v1.5 → models/bge-large-en-v1.5/
  2. direct HF first; on failure retry via hf-mirror.com (unless HF_ENDPOINT set)
  3. if neither <dir>/model.onnx nor <dir>/onnx/model.onnx exists, convert with
     optimum.onnxruntime (embedder accepts either layout)

Prereqs (only needed when actually downloading a model):
    <python> -m pip install huggingface_hub optimum[onnxruntime]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "BAAI/bge-large-en-v1.5"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "bge-large-en-v1.5"
MIRROR = "https://hf-mirror.com"


def onnx_present(model_dir: Path) -> bool:
    return (model_dir / "model.onnx").is_file() or (
        model_dir / "onnx" / "model.onnx"
    ).is_file()


def _snapshot(model_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"HF endpoint: {endpoint}")
    print(f"Downloading {MODEL_ID} -> {model_dir}")
    try:
        snapshot_download(MODEL_ID, local_dir=str(model_dir))
    except Exception as exc:  # noqa: BLE001 — retry via mirror, then re-raise
        if endpoint == MIRROR:
            raise
        print(f"direct download failed ({exc}); retrying via {MIRROR}", file=sys.stderr)
        os.environ["HF_ENDPOINT"] = MIRROR
        snapshot_download(MODEL_ID, local_dir=str(model_dir))


def _convert_onnx(model_dir: Path) -> None:
    print("Converting to ONNX format (optimum)...")
    try:
        import optimum  # noqa: F401 — presence check before shelling out
    except ImportError:
        print(
            "optimum is not installed. Run: "
            f'"{sys.executable}" -m pip install optimum[onnxruntime]',
            file=sys.stderr,
        )
        sys.exit(1)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "optimum.onnxruntime",
            "--model",
            str(model_dir),
            "--task",
            "feature-extraction",
            str(model_dir),
        ],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="target directory (default: <repo>/models/bge-large-en-v1.5)",
    )
    args = ap.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        _snapshot(model_dir)
        if not onnx_present(model_dir):
            _convert_onnx(model_dir)
    except Exception as exc:  # noqa: BLE001 — actionable message, non-zero exit
        print(f"model setup failed: {exc}", file=sys.stderr)
        return 1

    if not onnx_present(model_dir):
        print(
            "download finished but no ONNX file found; "
            f"expected {model_dir}/onnx/model.onnx or {model_dir}/model.onnx",
            file=sys.stderr,
        )
        return 1

    print(f"Model ready at {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
