
"""Basic checks for generated Instamart project outputs."""

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"


def test_required_reports_exist():
    required = [
        "audit.json",
        "sql_analysis_summary.json",
        "customer_segments.csv",
        "model_metrics.json",
        "triage_metrics.json",
        "policy_eval_results.csv",
        "agent_eval_results.csv",
    ]

    for filename in required:
        assert (REPORTS / filename).exists(), (
            f"Missing report: {filename}"
        )


def test_ml_f1_score():
    metrics = json.loads(
        (REPORTS / "model_metrics.json").read_text()
    )
    assert metrics["f1"] >= 0.55


def test_sql_cross_check():
    summary = json.loads(
        (REPORTS / "sql_analysis_summary.json").read_text()
    )
    check = summary["pandas_cross_check"]

    assert check["count_match"] is True
    assert check["revenue_match"] is True


def test_customer_segments():
    df = pd.read_csv(REPORTS / "customer_segments.csv")

    segment_col = next(
        col for col in df.columns
        if "segment" in col.lower() or "cluster" in col.lower()
    )

    assert df[segment_col].nunique() == 4
