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
    df = pd.read_csv(CSV_PATH)

    classes = sorted(df["gt_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_true = df["gt_label"].map(class_to_idx).values
    conf = df["confidence"].values
    pred_idx = df["pred_label"].map(class_to_idx).values

    logits = np.zeros((len(conf), len(classes)))
    for i in range(len(conf)):
        p = np.clip(conf[i], 1e-6, 1 - 1e-6)
        logits[i, pred_idx[i]] = np.log(p / (1 - p))

    res = minimize(
        lambda t: nll_loss(t[0], logits, y_true),
        x0=[1.0],
        bounds=[(0.05, 10.0)],
    )

    T = float(res.x[0])
    OUT_PATH.write_text(f"{T:.4f}")

    print(f"[OK] Temperature scaling T = {T:.4f}")
    print(f"[SAVED] {OUT_PATH}")


if __name__ == "__main__":
    main()
