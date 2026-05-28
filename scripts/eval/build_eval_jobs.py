#!/usr/bin/env python3
"""Build per-job eval configs and a command manifest from configs/eval/submit/jobs.yaml."""
from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_JOBS_CONFIG = Path("configs/eval/submit/jobs.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/submit")

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top-level in {path}")
    return data


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-._")
    return cleaned or "job"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def build_jobs(jobs_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    spec = _load_yaml(jobs_path)
    base_dir = jobs_path.parent

    # --- base config ---
    base_config_path = (base_dir / str(spec.get("base_config", "../default.yaml"))).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"base_config not found: {base_config_path}")
    base_config = _load_yaml(base_config_path)

    # --- resolve imports (models.yaml, benchmark_sets.yaml, etc.) ---
    for import_file in spec.get("imports", []):
        imported = _load_yaml((base_dir / import_file).resolve())
        for key in ("models", "benchmark_sets"):
            if key in imported:
                spec.setdefault(key, {}).update(imported[key])

    # --- profiles (inline or merged from imports) ---
    model_defs: dict[str, Any] = spec.get("models", {})
    benchmark_sets: dict[str, Any] = spec.get("benchmark_sets", {})
    if not model_defs:
        raise ValueError("No models defined — check jobs.yaml or imported models.yaml")
    if not benchmark_sets:
        raise ValueError("No benchmark_sets defined — check jobs.yaml or imported benchmark_sets.yaml")

    defaults: dict[str, Any] = spec.get("defaults", {}) or {}
    jobs: list[dict[str, Any]] = spec.get("jobs", [])
    if not jobs:
        raise ValueError("'jobs' list is empty — nothing to build")

    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    # Expand seed_range entries into multiple jobs before processing.
    expanded_jobs: list[dict[str, Any]] = []
    for job in jobs:
        seed_range = job.get("seed_range")
        if seed_range is not None:
            start = int(seed_range.get("start", 0))
            count = int(seed_range.get("count", 1))
            for s in range(start, start + count):
                j = deepcopy(job)
                j.pop("seed_range")
                j["name"] = f"{j.get('name', 'job')}-seed{s}"
                j.setdefault("config_overrides", {}).setdefault("evaluation", {})["seed"] = s
                expanded_jobs.append(j)
        else:
            expanded_jobs.append(job)
    jobs = expanded_jobs

    for i, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{i}] must be a mapping")

        # --- resolve model ---
        model_key = job.get("model") or defaults.get("model")
        if model_key not in model_defs:
            raise ValueError(f"jobs[{i}]: unknown model {model_key!r} (defined: {list(model_defs)})")
        model = model_defs[model_key]

        # --- resolve benchmark set ---
        bm_key = job.get("benchmarks") or defaults.get("benchmarks")
        if bm_key not in benchmark_sets:
            raise ValueError(f"jobs[{i}]: unknown benchmark_set {bm_key!r} (defined: {list(benchmark_sets)})")
        bm_list = benchmark_sets[bm_key]
        if isinstance(bm_list, str):
            bm_list = [b.strip() for b in bm_list.split(",") if b.strip()]
        if not bm_list:
            raise ValueError(f"jobs[{i}]: benchmark_set {bm_key!r} is empty")

        # --- tensor_parallel_size: single source of truth in the model definition ---
        tp_size = model.get("tensor_parallel_size")
        if tp_size is None:
            raise ValueError(
                f"Model {model_key!r} is missing 'tensor_parallel_size'. "
                "Set it in the models section of jobs.yaml — it controls both "
                "inference.tensor_parallel_size and the suggested GPU count for that job."
            )
        tp_size = int(tp_size)

        # --- build per-job config ---
        cfg = deepcopy(base_config)
        cfg["models"] = [{
            "name":           model_key,
            "base_model":     model.get("base_model"),
            "adapter_path":   model.get("adapter_path"),
            "adapter_name":   model.get("adapter_name"),
            "system_prompt":  model.get("system_prompt"),
        }]
        cfg["enabled_benchmarks"] = list(bm_list)

        # Inject per-model inference settings (all sourced from jobs.yaml models block).
        # temperature, top_p, and top_k go into inference (passed as GenerateConfig args).
        # pad_token_id is omitted — the tokenizer config defines it and vLLM reads it
        # automatically; passing it as a server arg breaks older vLLM versions.
        inf = cfg.setdefault("inference", {})
        inf["tensor_parallel_size"] = tp_size
        for key in ("temperature", "top_p", "top_k", "extra_body", "max_tokens"):
            if model.get(key) is not None:
                inf[key] = model[key]
        if model.get("vllm_server_args"):
            inf["vllm_server_args"] = _deep_merge(
                inf.get("vllm_server_args", {}), model["vllm_server_args"]
            )

        # optional raw overrides (escape hatch for advanced use); deep-merged so
        # nested keys (e.g. benchmarks.gpqa.limit) don't wipe sibling settings.
        if "config_overrides" in job:
            overrides = job["config_overrides"]
            if not isinstance(overrides, dict):
                raise ValueError(f"jobs[{i}].config_overrides must be a mapping")
            cfg = _deep_merge(cfg, overrides)
            # config_overrides.model (singular) merges into cfg["model"], but the
            # runner reads cfg["models"][0].  Propagate model-level overrides there.
            if "model" in overrides and cfg.get("models"):
                cfg["models"][0] = _deep_merge(cfg["models"][0], overrides["model"])

        # --- write generated config ---
        job_name = _sanitize(str(job.get("name") or f"{model_key}-{bm_key}"))
        meta = cfg.setdefault("_meta", {})
        # Store job_name in the config so inspect_runner can use it as the run folder slug.
        meta["job_name"] = job_name
        # Preserve the benchmark-set key from jobs.yaml so run/log directory names
        # can stay short (e.g. "all", "safety") instead of expanding every benchmark.
        meta["benchmark_set"] = str(bm_key)
        config_path = generated_dir / f"{job_name}.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        venv_path = model.get("venv_path") or defaults.get("venv_path") or "UNDEFINED"
        gpu_type = str(model.get("gpu_type") or "h100").strip().lower()

        row = {
            "job_name":       job_name,
            "config_path":    str(config_path),
            "suggested_gpus": str(tp_size),
            "venv_path":      str(venv_path).strip(),
            "gpu_type":       gpu_type,
            "command":        f"python -m src.cli --mode eval --config {config_path}",
        }
        rows.append(row)

    commands_path = output_dir / "eval_commands.sh"
    with commands_path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Generated by scripts/eval/build_eval_jobs.py\n")
        f.write("# Adapt these commands to your local or cluster execution pipeline.\n")
        for row in rows:
            f.write(
                "\n"
                f"# job_name: {row['job_name']}\n"
                f"# venv_path: {row['venv_path']}\n"
                f"# suggested_gpus: {row['suggested_gpus']}\n"
                f"# gpu_type: {row['gpu_type']}\n"
                f"{row['command']}\n"
            )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build eval configs and a command manifest.")
    parser.add_argument("--jobs-config", type=Path, default=DEFAULT_JOBS_CONFIG,
                        help=f"Path to jobs.yaml (default: {DEFAULT_JOBS_CONFIG})")
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    jobs_path  = args.jobs_config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_jobs(jobs_path, output_dir)
    print(
        f"Built {len(rows)} job(s): generated configs in "
        f"{output_dir / 'generated_configs'} and commands in {output_dir / 'eval_commands.sh'}"
    )
    for r in rows:
        print(
            f"  {r['job_name']:50s} "
            f"gpus={r['suggested_gpus']} "
            f"venv={r['venv_path']} "
            f"gpu_type={r['gpu_type']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
