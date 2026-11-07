
"""Run complete offline Instamart Copilot capstone."""

import json
import subprocess
import sys
from pathlib import Path

from src.foundation import build
from src.sql_analysis import run as sql_analysis
from src.segments import run as segments
from src.triage import run as triage
from src.policy import evaluate as policy
from src.agent import evaluate as agent


if __name__ == "__main__":

    # Step 1: Generate synthetic data
    subprocess.run(
        [sys.executable, "generate_data.py"],
        check=True,
        cwd=Path(__file__).resolve().parent
    )

    # Step 2: Run all project modules
    modules = [
        ("foundation", build),
        ("sql_analysis", sql_analysis),
        ("segments", segments),
        ("triage", triage),
        ("policy", policy),
        ("agent", agent),
    ]

    for name, function in modules:
        result = function()
        print(name, json.dumps(result, default=str))
