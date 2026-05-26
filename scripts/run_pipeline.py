#!/usr/bin/env python3
"""Run FMG pipeline steps 01–08."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import (
    step01_mrgal,
    step02_sample_prep,
    step03_modeling,
    step04_validation,
    step05_recommendation,
    step06_rank_mix,
    step07_lol,
    step08_so_output,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from-step", type=int, default=1)
    p.add_argument("--to-step", type=int, default=8)
    p.add_argument("--sample-n", type=int, default=50_000, help="Rows for ML steps (full data for 01-02)")
    p.add_argument(
        "--use-uc",
        action="store_true",
        help="Force Unity Catalog Delta I/O (requires Databricks cluster)",
    )
    args = p.parse_args()

    if args.use_uc:
        import os

        os.environ["FMG_USE_UNITY_CATALOG"] = "true"
        from src.databricks.bootstrap import bootstrap_notebook

        bootstrap_notebook(ROOT)

    steps = {
        1: ("MRGAL", lambda: step01_mrgal.run(sample_n=None)),
        2: ("Sample prep", lambda: step02_sample_prep.run(sample_n=None)),
        3: ("Modeling", lambda: step03_modeling.run(sample_n=args.sample_n)),
        4: ("Validation", lambda: step04_validation.run(sample_n=args.sample_n)),
        5: ("Recommendation", lambda: step05_recommendation.run()),
        6: ("Rank-mix", lambda: step06_rank_mix.run(sample_n=args.sample_n)),
        7: ("LOL", lambda: step07_lol.run()),
        8: ("SO output", lambda: step08_so_output.run()),
    }

    for n in range(args.from_step, args.to_step + 1):
        name, fn = steps[n]
        print(f"=== Step {n}: {name} ===")
        result = fn()
        print(result)


if __name__ == "__main__":
    main()
