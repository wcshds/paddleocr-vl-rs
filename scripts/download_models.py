#!/usr/bin/env python3
"""Download PaddleOCR-VL model files for the Rust examples.

The Rust crates expect this local layout by default:

  model1.0/
    model.safetensors
    tokenizer.json
    preprocessor_config.json
    PP-DocLayoutV2/model.safetensors

  model1.5/
    model.safetensors
    tokenizer.json
    preprocessor_config.json
    PP-DocLayoutV3/model.safetensors

Only the files needed by the Rust loaders are downloaded. Paddle inference files
such as `inference.pdiparams` are intentionally skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:  # pragma: no cover - this is a user-facing setup error.
    print(
        "Missing dependency: huggingface_hub\n"
        "Install it with: python -m pip install huggingface_hub",
        file=sys.stderr,
    )
    raise SystemExit(1)


DEFAULT_V1_REPO = "PaddlePaddle/PaddleOCR-VL"
DEFAULT_V1_5_REPO = "PaddlePaddle/PaddleOCR-VL-1.5"
DEFAULT_V2_LAYOUT_REPO = "PaddlePaddle/PP-DocLayoutV2_safetensors"
DEFAULT_V3_LAYOUT_REPO = "PaddlePaddle/PP-DocLayoutV3_safetensors"

VL_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "LICENSE*",
    "README*",
    "chat_template.jinja",
    "model.safetensors",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
]

def snapshot(repo_id: str, target_dir: Path, allow_patterns: list[str], revision: str | None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=target_dir,
        allow_patterns=allow_patterns,
    )


def download_single_file(
    repo_id: str,
    filename: str,
    target_path: Path,
    revision: str | None,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}/{filename} -> {target_path}")
    cached = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        filename=filename,
    )
    shutil.copy2(cached, target_path)


def download_v1(args: argparse.Namespace) -> None:
    root = args.output / "model1.0"
    snapshot(args.v1_repo, root, VL_ALLOW_PATTERNS, args.revision)
    download_single_file(
        args.layout_v2_repo,
        "model.safetensors",
        root / "PP-DocLayoutV2" / "model.safetensors",
        args.layout_v2_revision,
    )


def download_v1_5(args: argparse.Namespace) -> None:
    root = args.output / "model1.5"
    snapshot(args.v1_5_repo, root, VL_ALLOW_PATTERNS, args.revision)
    download_single_file(
        args.layout_v3_repo,
        "model.safetensors",
        root / "PP-DocLayoutV3" / "model.safetensors",
        args.layout_v3_revision,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PaddleOCR-VL checkpoints for paddleocr-vl-rs."
    )
    parser.add_argument(
        "--version",
        choices=["all", "1.0", "1.5"],
        default="all",
        help="which model set to download (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="output directory; model1.0/model1.5 are created under it",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="optional Hugging Face revision for PaddleOCR-VL repos",
    )
    parser.add_argument("--v1-repo", default=DEFAULT_V1_REPO)
    parser.add_argument("--v1-5-repo", default=DEFAULT_V1_5_REPO)
    parser.add_argument("--layout-v2-repo", default=DEFAULT_V2_LAYOUT_REPO)
    parser.add_argument("--layout-v3-repo", default=DEFAULT_V3_LAYOUT_REPO)
    parser.add_argument(
        "--layout-v2-revision",
        default=None,
        help="optional revision for PP-DocLayoutV2_safetensors",
    )
    parser.add_argument(
        "--layout-v3-revision",
        default=None,
        help="optional revision for PP-DocLayoutV3_safetensors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()

    if args.version in {"all", "1.0"}:
        download_v1(args)
    if args.version in {"all", "1.5"}:
        download_v1_5(args)

    print("\nDone. Default model paths are ready for the Rust APIs and CLI examples.")


if __name__ == "__main__":
    main()
