# Weather-Aware Perception Pipeline for Autonomous Driving

## Overview
This project implements an end-to-end computer vision pipeline
for road-scene understanding under varying weather conditions.

The pipeline performs:
1. Weather classification from a single road image
2. Object detection using a YOLO-based detector
3. Risk-aware fusion of weather and detection outputs

The system is designed to study robustness and perception behavior
in non-ideal environmental conditions.

---

## Pipeline Components

### Weather Classifier
- Architecture: ResNet18
- Training: Fine-tuned on weather-labeled road images
- Validation Accuracy: ~70%
- Output: Weather label + confidence

### Object Detector
- Model: YOLOv8s
- Format: ONNX (frozen)
- Task: Multi-class object detection on road scenes

### Fusion Logic
- Type: Rule-based (intentionally simple)
- Purpose: Combine weather and object information into a
  qualitative risk profile
- Design: Replaceable by learning-based fusion later

---

## Example Output

```json
{
  "weather": {
    "label": "overcast",
    "confidence": 0.39
  },
  "objects": [
    {"class": "car", "confidence": 0.93}
  ],
  "risk_profile": {
    "visibility": "moderate",
    "weather_confidence": 0.39,
    "num_objects": 20
  }
}

How to Run

Activate the virtual environment and run:

python -m code.pipeline.main

The pipeline expects a road-scene image path defined inside
code/pipeline/main.py.
Project Motivation

Most perception systems are evaluated under ideal conditions.
This project focuses on understanding perception behavior under
uncertain and adverse weather scenarios.

The goal is not maximum accuracy, but system-level robustness
and clean pipeline design.


