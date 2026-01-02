from pathlib import Path
import cv2
import numpy as np

# IMPORTANT: run this file ONLY with
# python -m analysis.decision_visualize

from code.pipeline.main import run_pipeline
from analysis.lane_visibility_metrics import compute_lane_metrics

# -----------------------------
# OUTPUT DIRECTORY
# -----------------------------
OUTPUT_DIR = Path("analysis/decision_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# VISUAL HELPERS
# -----------------------------
def overlay_lane(image, mask):
    """
    Overlay lane mask in green on the image.
    mask must be binary (H x W).
    """
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :, 1] = 255  # green channel

    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(
        overlay[mask_bool], 0.5, green[mask_bool], 0.5, 0
    )
    return overlay


def draw_text_block(img, lines, x=20, y=40):
    """
    Draw multiple lines of text on image.
    """
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


# -----------------------------
# MAIN VISUALIZATION LOGIC
# -----------------------------
def visualize(image_path: str):
    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"

    # Load image
    img = cv2.imread(str(image_path))
    assert img is not None, f"Failed to load image: {image_path}"

    img = cv2.resize(img, (1280, 720))

    # -------------------------
    # RUN FULL PIPELINE
    # -------------------------
    output = run_pipeline(str(image_path))

    # -------------------------
    # LANE METRICS
    # -------------------------
    lane_metrics = compute_lane_metrics(
        np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    )

    # NOTE:
    # We dropped UNet.
    # Lane visibility comes from UFLD (lane_count + rules),
    # so we DO NOT compute a pixel mask anymore.

    # -------------------------
    # TEXT BLOCK
    # -------------------------
    text = [
        f"Weather: {output['weather']['label']} ({output['weather']['confidence']:.2f})",
        f"Detected objects: {len(output['detections'])}",
        f"Lane count: {output['lane_count']}",
        f"Final visibility: {output['risk_profile']['visibility']}",
    ]

    draw_text_block(img, text)

    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    out_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(out_path), img)
    print(f"[SAVED] {out_path}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    # ADD OR REMOVE IMAGES HERE ONLY
    images = [
        "results/showcase/00a2e3ca-5c856cde.jpg",
        # "results/showcase/example2.jpg",
        # "results/showcase/example3.jpg",
    ]

    for img in images:
        visualize(img)
