# analysis/failure_analysis.py

import pandas as pd
from pathlib import Path

CSV_PATH = Path("analysis/decision_debug/summary.csv")
OUT_DIR = Path("analysis/failure_reports")
OUT_DIR.mkdir(exist_ok=True)


def main():
    assert CSV_PATH.exists(), "summary.csv missing"

    df = pd.read_csv(CSV_PATH)

    required = {
        "gt_weather",
        "pred_weather",
        "confidence",
        "trusted",
        "correct",
    }

    missing = required - set(df.columns)
    assert not missing, f"CSV missing columns: {missing}"

    # -------------------------
    # Normalize missing values
    # -------------------------
    df["trusted"] = df["trusted"].fillna(False)

    # -------------------------
    # Accuracy by weather
    # -------------------------
    df.groupby("gt_weather")["correct"].mean().to_csv(
        OUT_DIR / "accuracy_by_weather.csv"
    )

    # -------------------------
    # Low-confidence cases
    # -------------------------
    df[df["confidence"] < 0.4].to_csv(
        OUT_DIR / "low_confidence.csv", index=False
    )

    # -------------------------
    # Untrusted decisions
    # -------------------------
    df[df["trusted"] == False].to_csv(
        OUT_DIR / "untrusted_decisions.csv", index=False
    )

    # -------------------------
    # Misclassifications
    # -------------------------
    df[df["correct"] == False].to_csv(
        OUT_DIR / "misclassified.csv", index=False
    )

    print("[DONE] Failure analysis complete")


if __name__ == "__main__":
    main()
