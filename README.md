# Weather-Aware Object Detection Pipeline

This project implements a multi-stage computer vision pipeline for
robust road-scene perception under varying environmental conditions.

## Overview
The system combines:
- A CNN-based weather classifier
- A YOLOv8 object detection model
- A unified inference pipeline with risk profiling

The goal is to explicitly model environmental context (weather)
and analyze object detection outputs under real-world conditions
such as overcast skies, rain, and reduced visibility.

## Pipeline
Image → Weather Estimation → Object Detection → Risk Profiling

## Demo
Below is an example output produced by the pipeline on a real road image:

![Demo](results/demo/demo.jpg)

## Notes
- Models are trained on large-scale real-world datasets
- The pipeline is designed for clarity, modularity, and extensibility
- Uncertainty in predictions is explicitly preserved
