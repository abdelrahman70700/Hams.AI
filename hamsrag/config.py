"""Project paths and YAML config loading."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("HAMS_ROOT", Path(__file__).resolve().parents[1]))


def load_config(name: str) -> dict:
    path = ROOT / "configs" / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def p(rel: str | Path) -> Path:
    """Resolve a project-relative path."""
    rel = Path(rel)
    return rel if rel.is_absolute() else ROOT / rel
