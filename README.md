Weather-Aware Lane Perception and Decision Pipeline


Overview

This project presents an end-to-end perception and decision pipeline for road scene understanding under varying weather conditions.
The system integrates weather classification, lane perception, object detection, and a confidence-aware decision policy, followed by automated failure analysis.
The primary objective is to study how environmental uncertainty (clear, rainy, snowy conditions) affects perception reliability and downstream decision-making.
The pipeline runs fully offline, produces structured outputs, and is designed for interpretability, robustness, and evaluation, rather than aggressive optimization.



Key Capabilities

Weather classification with confidence estimation
Lane perception and visibility assessment
Object detection for contextual awareness
Confidence-aware decision policy
Automated failure and reliability analysis
Reproducible demo and evaluation outputs


System Architecture
Input Image / Video
        │
        ▼
Weather Classification (ResNet-18)
        │
        ├───────────────┐
        │               │
        ▼               ▼
Lane Perception      Object Detection
(UFLD / UNet)        (YOLOv8)
        │               │
        └───────┬───────┘
                ▼
      Confidence-Aware Fusion
                │
                ▼
        Decision Policy Layer
                │
                ▼
   Structured Outputs + Analysis


Note:
Object detection using YOLOv8 is incorporated as an auxiliary perception module.
Detected objects contribute to contextual awareness and fusion logic, while primary decision-making is driven by weather confidence and lane visibility.


Project Structure

Project/
│
├── analysis/                 # Evaluation, visualization, failure analysis
│   ├── decision_debug/
│   ├── failure_reports/
│   ├── decision_engine.py
│   ├── decision_logic.py
│   ├── decision_visualize.py
│   └── failure_analysis.py
│
├── code/                     # Core pipeline
│   ├── classifier/           # Weather classification
│   ├── detector/             # Object detection (YOLO)
│   ├── pipeline/             # Perception + decision logic
│   ├── unet/                 # Lane segmentation
│   ├── utils/
│   └── yolo/
│
├── models/                   # Pretrained weights
│   ├── weather/
│   ├── unet/
│   └── yolo/
│
├── results/                  # Demo outputs (videos )
│
├── datasets/                 # Demo videos
├── tools/                    # Utility scripts
├── README.md
└── .gitignore



Models Used
Component	Model
Weather Classification	ResNet-18 (fine-tuned)
Lane Detection	UFLD
Lane Segmentation	UNet
Object Detection	YOLOv8
Decision Policy


The decision layer is explicit and rule-based, designed for transparency:

Low weather confidence → untrusted decisions
Poor lane visibility → cautious or slow mode
Adverse weather (rain/snow) → conservative behavior
Clear conditions with reliable perception → normal mode

Each decision includes:
Mode (normal, slow, caution)
Trust flag
Reason code


How to Run
1. Create virtual environment
python -m venv venv

2. Activate environment

Windows

venv\Scripts\activate

3. Install dependencies
pip install torch torchvision opencv-python numpy pandas ultralytics addict

4. Run visualization + pipeline
python -m analysis.decision_visualize


This generates:

analysis/decision_debug/summary.csv

5. Run failure analysis
python -m analysis.failure_analysis



Outputs:

analysis/failure_reports/
├── accuracy_by_weather.csv
├── low_confidence.csv
├── misclassified.csv
└── untrusted_decisions.csv



Outputs

Primary Outputs

Predicted weather label with confidence
Lane visibility assessment
Decision mode and trust flag
Annotated demo video

Analysis Outputs

Accuracy by weather condition
Misclassification cases
Low-confidence predictions
Untrusted decision samples



Current Performance (Qualitative)

Clear: High reliability
Rainy: Moderate reliability
Snowy: Lower reliability (expected due to visual ambiguity)

The system prioritizes robustness and interpretability over raw accuracy.



Design Principles

Modular pipeline architecture
Single source of truth for decisions
Explicit confidence handling
Clear separation between inference and analysis
Reproducible and inspectable outputs



Status

Pipeline complete
Models integrated
Analysis automated
Demo outputs generated
Ready for academic review and presentation



Notes

Training scripts and experimental code are intentionally excluded
The project focuses on system integration, not dataset-specific optimization
Outputs are intended for demonstration and analysis, not real-time deployment


Author
Utkarsh