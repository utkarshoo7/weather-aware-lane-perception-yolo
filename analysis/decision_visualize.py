from pathlib import Path
import cv2
import numpy as np

from code.pipeline.main import run_pipeline
from code.unet.infer import predict_lane_mask
from analysis.lane_visibility_metrics import compute_lane_metrics


OUTPUT_DIR = Path("analysis/decision_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def overlay_lane(image, mask):
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :, 1] = 255

    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(
        overlay[mask_bool], 0.5, green[mask_bool], 0.5, 0
    )
    return overlay


def draw_text_block(img, lines, x=20, y=40):
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x, y + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def visualize(image_path: str):
    image_path = Path(image_path)
    assert image_path.exists(), image_path

    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (1280, 720))

    # Run pipeline
    output = run_pipeline(str(image_path))

    # Lane mask
    lane_mask = predict_lane_mask(str(image_path))

    # Lane metrics
    lane_metrics = compute_lane_metrics(lane_mask)

    # Overlay lanes
    vis = overlay_lane(img, lane_mask)

    # Text info
    text = [
        f"Weather: {output['weather']['label']} ({output['weather']['confidence']:.2f})",
        f"Lane pixel ratio: {lane_metrics['lane_pixel_ratio']:.4f}",
        f"Bottom density: {lane_metrics['bottom_half_density']:.4f}",
        f"Vertical continuity: {lane_metrics['vertical_continuity']:.4f}",
        f"Final visibility: {output['risk_profile']['visibility']}",
    ]

    draw_text_block(vis, text)

    # Save
    out_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(out_path), vis)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    # CHANGE ONLY THESE IMAGE PATHS
    images = [
        "results/showcase/00a2e3ca-5c856cde.jpg",
    ]

    for img in images:
        visualize(img)
