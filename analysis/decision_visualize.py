# analysis/decision_visualize.py

from pathlib import Path
import csv

from code.pipeline.main import run_pipeline

SHOWCASE_DIR = Path("results/showcase")
OUT_DIR = Path("analysis/decision_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = []

    for gt_dir in sorted(SHOWCASE_DIR.iterdir()):
        if not gt_dir.is_dir():
            continue

        gt_weather = gt_dir.name

        for img_path in gt_dir.glob("*.jpg"):
            result = run_pipeline(str(img_path))

            row = {
                "image": img_path.name,
                "gt_weather": gt_weather,
                "pred_weather": result["weather"]["label"],
                "confidence": result["weather"]["confidence"],
                "lane_visibility": result["lane"]["visibility"],
                "lane_count": result["lane"]["count"],
                # SAFE ACCESS (no KeyError)
                "trusted": result["decision"].get("trusted"),
                "mode": result["decision"].get("mode"),
                "correct": gt_weather == result["weather"]["label"],
            }

            rows.append(row)

    if not rows:
        raise RuntimeError("No images found for decision visualization")

    out_csv = OUT_DIR / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
