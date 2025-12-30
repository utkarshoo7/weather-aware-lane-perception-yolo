import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("analysis/weather_calibration_data.csv")
OUT_PATH = Path("analysis/weather_reliability.png")

N_BINS = 10


def main():
    assert CSV_PATH.exists(), f"Missing {CSV_PATH}"

    df = pd.read_csv(CSV_PATH)

    # Required columns
    required = {"confidence", "gt_label", "pred_label"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    confidences = df["confidence"].values
    correct = (df["gt_label"] == df["pred_label"]).astype(int).values

    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    acc_per_bin = []
    conf_per_bin = []

    for i in range(N_BINS):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            acc_per_bin.append(0)
            conf_per_bin.append(0)
        else:
            acc_per_bin.append(correct[mask].mean())
            conf_per_bin.append(confidences[mask].mean())

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.bar(
        bin_centers,
        acc_per_bin,
        width=0.08,
        edgecolor="black",
        alpha=0.7,
        label="Empirical Accuracy",
    )
    plt.plot(
        bin_centers,
        conf_per_bin,
        "ro-",
        label="Mean Confidence",
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Weather Classifier Reliability Diagram")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUT_PATH)
    plt.close()

    print("\n=== RELIABILITY DIAGRAM SAVED ===")
    print(f"Output : {OUT_PATH}")


if __name__ == "__main__":
    main()
