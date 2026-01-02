from pathlib import Path
import csv

from code.classifier.infer import predict_weather

DATASET_ROOT = Path("datasets/weather/val")
OUT_FILE = Path("analysis/weather_calibration_data.csv")

CLASSES = ["clear", "overcast", "partly_cloudy", "rainy", "snowy"]


def main():
    rows = []
    total = 0

    print("\n=== SCANNING WEATHER VALIDATION SET ===")

    for gt_label in CLASSES:
        class_dir = DATASET_ROOT / gt_label
        assert class_dir.exists(), f"Missing folder: {class_dir}"

        images = list(class_dir.glob("*.jpg"))
        print(f"{gt_label}: {len(images)} images")

        for img in images:
            pred_label, confidence = predict_weather(str(img))

            rows.append({
                "image": str(img),
                "gt_label": gt_label,
                "pred_label": pred_label,
                "confidence": float(confidence),
            })

            total += 1

    OUT_FILE.parent.mkdir(exist_ok=True)

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "gt_label", "pred_label", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== WEATHER CALIBRATION DATA SAVED ===")
    print(f"Samples : {total}")
    print(f"Output  : {OUT_FILE}")


if __name__ == "__main__":
    main()
