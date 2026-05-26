from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def load_settings() -> dict[str, Any]:
    path = _ROOT / "config" / "settings.yaml"
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["project_root"] = str(_ROOT)
    row = os.environ.get("FMG_ROW_COUNT")
    if row:
        cfg["synthetic"]["row_count"] = int(row)
    client = os.environ.get("FMG_CLIENT_ID")
    if client:
        cfg["client"]["id"] = client
    if os.environ.get("FMG_ALLOW_SYNTHETIC_RESPONDERS", "").lower() in ("1", "true", "yes"):
        cfg.setdefault("sample_prep", {})["allow_synthetic_responders"] = True
    return cfg


def project_root() -> Path:
    return Path(load_settings()["paths"]["project_root"])


def resolve_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)
