# analysis/weather_confidence_calibration.py

import pandas as pd
import numpy as np
from pathlib import Path

N_BINS = 10
DATA_PATH = Path("analysis/weather_calibration_data.csv")


def compute_ece(confidence, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidence > bins[i]) & (confidence <= bins[i + 1])
        if mask.sum() == 0:
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        ece += (mask.sum() / len(confidence)) * abs(bin_acc - bin_conf)

    return ece


def main():
    df = pd.read_csv(DATA_PATH)

    confidence = df["confidence"].values
    correct = (df["gt_label"] == df["pred_label"]).astype(int).values

    ece = compute_ece(confidence, correct, N_BINS)

    print("\n=== WEATHER CONFIDENCE CALIBRATION ===")
    print(f"Samples : {len(df)}")
    print(f"ECE     : {ece:.4f}")

    out = Path("analysis/weather_ece.txt")
    out.write_text(f"{ece:.6f}")

    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
