"""
Resolve GROQ API key without hardcoding in committed source.

Priority:
  1. Environment variable GROQ_API_KEY
  2. config/groq.local.yaml (gitignored)
  3. src/groq_secrets.local.py (gitignored) — paste key here for local / Databricks Repos
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def get_groq_api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    local_yaml = _ROOT / "config" / "groq.local.yaml"
    if local_yaml.exists():
        import yaml

        with local_yaml.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        key = str(data.get("api_key", "")).strip()
        if key:
            os.environ.setdefault("GROQ_API_KEY", key)
            return key

    secrets_py = Path(__file__).resolve().parent / "groq_secrets.local.py"
    if secrets_py.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("groq_secrets_local", secrets_py)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            key = str(getattr(mod, "GROQ_API_KEY", "")).strip()
            if key:
                os.environ.setdefault("GROQ_API_KEY", key)
                return key

    return None
