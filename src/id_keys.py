"""Individual ID (indiv_id) assignment and shared merge keys."""

from __future__ import annotations

import pandas as pd

PERSON_KEY = "indiv_id"
LEGACY_PERSON_KEY = "bpid"
PERSON_ID_ALIASES = ("individual_id", "inidivid_id", "individ_id")

KEY_COLUMNS = [
    LEGACY_PERSON_KEY,
    PERSON_KEY,
    "campaign_id",
    "client_id",
    "cut_date",
    "product_code",
]

MODELING_EXCLUDE = {
    LEGACY_PERSON_KEY,
    PERSON_KEY,
    "campaign_id",
    "client_id",
    "cut_date",
    "product_code",
    "responder_flag",
    "premium_amount",
}


def ensure_indiv_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure indiv_id exists; map from aliases or bpid when missing."""
    if PERSON_KEY in df.columns and df[PERSON_KEY].notna().all():
        return df
    out = df.copy()
    if PERSON_KEY not in out.columns or out[PERSON_KEY].isna().any():
        assigned = False
        for alt in PERSON_ID_ALIASES:
            if alt in out.columns:
                out[PERSON_KEY] = out[alt]
                assigned = True
                break
        if not assigned:
            if LEGACY_PERSON_KEY in out.columns:
                out[PERSON_KEY] = out[LEGACY_PERSON_KEY]
            else:
                raise ValueError(f"Cannot assign {PERSON_KEY}: need bpid or individual_id column")
    if LEGACY_PERSON_KEY not in out.columns:
        out[LEGACY_PERSON_KEY] = out[PERSON_KEY]
    return out
