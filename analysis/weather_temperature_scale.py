import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

CSV_PATH = Path("analysis/weather_calibration_data.csv")
OUT_PATH = Path("analysis/weather_temperature.txt")


def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def nll_loss(T, logits, labels):
    scaled_logits = logits / T
    probs = softmax(scaled_logits)
    eps = 1e-12
    return -np.mean(np.log(probs[np.arange(len(labels)), labels] + eps))


def main():
    assert CSV_PATH.exists(), "Missing calibration CSV"

    df = pd.read_csv(CSV_PATH)

    # Encode labels
    classes = sorted(df["gt_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_true = df["gt_label"].map(class_to_idx).values
    conf = df["confidence"].values

    # Reconstruct pseudo-logits from confidence
    # log(p / (1-p)) for predicted class
    logits = np.zeros((len(conf), len(classes)))
    pred_idx = df["pred_label"].map(class_to_idx).values

    for i in range(len(conf)):
        p = np.clip(conf[i], 1e-6, 1 - 1e-6)
        logits[i, pred_idx[i]] = np.log(p / (1 - p))

    # Optimize temperature
    res = minimize(
        lambda t: nll_loss(t[0], logits, y_true),
        x0=[1.0],
        bounds=[(0.05, 10.0)],
    )

    T = float(res.x[0])
    OUT_PATH.write_text(f"{T:.4f}")

    print("\n=== TEMPERATURE SCALING COMPLETE ===")
    print(f"Optimal T : {T:.4f}")
    print(f"Saved to  : {OUT_PATH}")


if __name__ == "__main__":
    main()
