#!/usr/bin/env python3
"""Generate Regional Bank synthetic bronze tables."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic_data import generate_regional_bank


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=None, help="Override row count (default settings.yaml)")
    p.add_argument("--no-prior", action="store_true", help="Skip prior-campaign bronze tables")
    args = p.parse_args()

    if args.rows:
        os.environ["FMG_ROW_COUNT"] = str(args.rows)

    print("Generating Regional Bank bronze data...")
    paths = generate_regional_bank(include_prior=not args.no_prior)
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print("Done.")


if __name__ == "__main__":
    main()
