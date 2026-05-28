#!/usr/bin/env python3
"""Standalone script to merge LoRA adapters into base models.

Usage:
    python scripts/merge/merge_lora_models.py MODEL_KEY [MODEL_KEY ...]

MODEL_KEY must match a key in configs/eval/submit/models.yaml that has a
non-null adapter_path. The merged model is saved under:
    outputs/merged_models/<model_key>/

Override the output root with --output-dir.

Example:
    python scripts/merge/merge_lora_models.py nemotron-traits nemotron-vs-50-50
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("MERGED_MODEL_DIR", REPO_ROOT / "outputs" / "merged_models")
)
MODELS_YAML = REPO_ROOT / "configs/eval/submit/models.yaml"


def load_models() -> dict:
    with open(MODELS_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_keys", nargs="+", metavar="MODEL_KEY", help="Model key(s) from models.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root directory for merged models")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--force", action="store_true", help="Re-merge even if output already exists")
    args = parser.parse_args()

    # Import merge function from the repo's source
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from runners.inspect_runner import merge_lora  # noqa: E402

    models = load_models()
    errors = []

    for key in args.model_keys:
        if key not in models:
            LOGGER.error("Unknown model key '%s'. Available keys: %s", key, ", ".join(sorted(models)))
            errors.append(key)
            continue

        cfg = models[key]
        adapter_path = cfg.get("adapter_path")
        if not adapter_path:
            LOGGER.error("Model '%s' has no adapter_path — nothing to merge.", key)
            errors.append(key)
            continue

        base_model = cfg["base_model"]
        output_dir = args.output_dir / key

        if not args.force and output_dir.exists() and (output_dir / "config.json").exists():
            LOGGER.info("Skipping '%s' — merged model already exists at %s (use --force to redo)", key, output_dir)
            continue

        LOGGER.info("Merging '%s': %s + %s -> %s", key, base_model, adapter_path, output_dir)
        try:
            merge_lora(base_model=base_model, adapter_path=adapter_path, output_dir=output_dir, dtype=args.dtype)
            LOGGER.info("Done: '%s' saved to %s", key, output_dir)
        except Exception as exc:
            LOGGER.error("Failed to merge '%s': %s", key, exc, exc_info=True)
            errors.append(key)

    if errors:
        LOGGER.error("The following models failed or were skipped due to errors: %s", errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
