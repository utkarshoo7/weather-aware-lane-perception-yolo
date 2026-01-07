"""
Video demo runner for Weather-Aware Perception Pipeline.

This script:
- Reads a video file
- Runs frame-wise inference using the stable pipeline
- Overlays weather + decision information
- Produces a clean demo video (no lane / YOLO visuals)

This file is for DEMONSTRATION ONLY.
"""

import cv2
from pathlib import Path
import tempfile

from code.pipeline.main import run_pipeline


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
INPUT_VIDEO = Path("datasets/demo_videos/clear.mp4")
OUTPUT_VIDEO = Path("results/demo_clear_out.mp4")

assert INPUT_VIDEO.exists(), f"Missing input video: {INPUT_VIDEO}"
OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# VIDEO SETUP
# ------------------------------------------------------------------
cap = cv2.VideoCapture(str(INPUT_VIDEO))
assert cap.isOpened(), "Failed to open video"

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))


# ------------------------------------------------------------------
# FRAME LOOP
# ------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame temporarily (pipeline expects path)
        frame_path = Path(tmpdir) / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        # Run pipeline
        result = run_pipeline(str(frame_path))

        # ----------------------------------------------------------
        # OVERLAY: WEATHER
        # ----------------------------------------------------------
        weather = result["weather"]["label"]
        conf = result["weather"]["confidence"]

        cv2.putText(
            frame,
            f"Weather: {weather} ({conf:.2f})",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------------
        # OVERLAY: DECISION
        # ----------------------------------------------------------
        decision = result["decision"]
        mode = decision["mode"]
        trusted = decision["trusted"]

        cv2.putText(
            frame,
            f"Mode: {mode} | Trusted: {trusted}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)
        frame_id += 1


# ------------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------------
cap.release()
out.release()

print(f"[DONE] Demo video saved to: {OUTPUT_VIDEO}")
