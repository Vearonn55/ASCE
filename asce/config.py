"""YAML pipeline configs. One file fully describes a pipeline, plus an optional train block."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

STAGE_KEYS = ("source", "preprocessor", "segmenter", "features", "model", "decoder")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must be a mapping")
    return raw


def stage_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    missing = [key for key in STAGE_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"config missing stage keys: {', '.join(missing)}")
    specs = {}
    for key in STAGE_KEYS:
        spec = cfg[key]
        if not isinstance(spec, dict):
            raise ValueError(f"stage '{key}' must be a mapping")
        specs[key] = spec
    return specs
