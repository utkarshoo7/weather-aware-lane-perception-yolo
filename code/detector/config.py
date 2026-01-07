"""
Object Detection Configuration
------------------------------
Centralized configuration for YOLO-based object detection.
This file contains only deployment-level constants.
"""

from pathlib import Path


# ==================================================
# MODEL CONFIGURATION
# ==================================================

# Pretrained YOLOv8 model exported to ONNX for inference
YOLO_MODEL_PATH = Path("models/yolo/best.onnx")

# Input resolution expected by the detector
IMAGE_SIZE = 640


# ==================================================
# INFERENCE THRESHOLDS
# ==================================================

# Minimum confidence score for a detection to be accepted
CONF_THRESHOLD = 0.25
