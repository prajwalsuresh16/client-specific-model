"""
Column schemas for FMG modeling files (SD, STAT, MRGAL, responders).
Regional Bank is the sole client for this POC.
"""

from __future__ import annotations

from src.id_keys import KEY_COLUMNS, ensure_indiv_id

__all__ = ["KEY_COLUMNS", "ensure_indiv_id", "SD_CORE", "REGIONAL_BANK_SD_COLUMNS"]

SD_CORE = [
    "eligible_to_market_flag",
    "deceased_suppressed_flag",
    "bad_address_flag",
    "dnc_flag",
    "existing_product_suppressed_flag",
    "household_id",
    "state_code",
    "zip_code",
    "age",
    "gender_code",
    "income_band",
    "marital_status_code",
    "homeowner_flag",
    "insured_flag",
    "max_coverage_reached_flag",
]

STAT_PROMO = [
    "promo_mail_count_12m",
    "promo_mail_count_36m",
    "last_promo_days_ago",
    "last_response_days_ago",
    "prior_product_count",
]

STAT_MEMBERSHIP = [
    "membership_tenure_months",
    "membership_tier_code",
    "account_open_days",
    "account_balance_band",
    "direct_deposit_flag",
]

STAT_DEMO = [
    "census_income_index",
    "wealth_score",
    "credit_risk_band",
    "life_stage_code",
    "urbanicity_code",
    "education_code",
    "household_size",
    "children_present_flag",
    "pet_owner_flag",
    "travel_affinity_score",
    "health_affinity_score",
    "financial_stress_index",
]

STAT_EXTENDED = [
    "loan_to_value_band",
    "deposit_balance_band",
    "transaction_velocity_score",
    "online_login_frequency_band",
    "branch_proximity_miles_band",
    "marketing_opt_in_email_flag",
    "marketing_opt_in_sms_flag",
]

STAT_COMMON = STAT_PROMO + STAT_MEMBERSHIP + STAT_DEMO + STAT_EXTENDED

INFOBASE_ATTRS = [f"infobase_attr_{i:03d}" for i in range(1, 16)]

MRGAL_FLAGS = [
    "news_p_flag",
    "news_d_flag",
    "base_name_flag",
    "prior_promo_depth_score",
]

REGIONAL_BANK_EXTRA = [
    "regional_branch_code",
    "digital_banking_user_flag",
    "cd_balance_band",
    "mortgage_holder_flag",
    "auto_loan_flag",
]

RESPONDER_COLS = [
    "responder_flag",
    "response_date",
    "premium_amount",
    "policy_issued_flag",
    "channel_code",
    "response_lag_days",
    "coverage_amount_band",
    "payment_mode_code",
    "cancel_within_30d_flag",
    "claim_filed_flag",
    "multi_product_responder_flag",
    "campaign_touch_index",
    "mail_piece_version",
    "creative_test_cell",
    "underwriting_decision_code",
    "agent_channel_flag",
    "digital_response_flag",
    "inbound_call_flag",
    "apps_started_count",
    "apps_completed_count",
    "household_responder_flag",
    "ltv_score_at_response",
    "risk_score_at_response",
    "discount_applied_flag",
    "renewal_probability_score",
    "cross_sell_eligible_flag",
    "actual_pppm",
    "actual_response_decile",
    "sales_book_actualization_pct",
    "maturity_days",
]

REGIONAL_BANK_SD_COLUMNS = KEY_COLUMNS + SD_CORE + INFOBASE_ATTRS + REGIONAL_BANK_EXTRA
REGIONAL_BANK_STAT_COLUMNS = KEY_COLUMNS + STAT_COMMON
REGIONAL_BANK_MRGL_COLUMNS = (
    KEY_COLUMNS + SD_CORE + STAT_COMMON + INFOBASE_ATTRS + MRGAL_FLAGS + REGIONAL_BANK_EXTRA
)
REGIONAL_BANK_RESPONDER_COLUMNS = KEY_COLUMNS + RESPONDER_COLS


def assert_min_columns(cols: list[str], minimum: int = 30, label: str = "") -> None:
    if len(cols) < minimum:
        raise ValueError(f"{label} has {len(cols)} columns; need at least {minimum}")
