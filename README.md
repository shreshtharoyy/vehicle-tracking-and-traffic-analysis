# Vehicle Tracking and Traffic Analytics System

A real-time computer vision pipeline for traffic monitoring, built with YOLOv8, OpenCV, and Supervision. Processes highway footage to deliver vehicle detection, multi-object tracking, flow analysis, speed estimation, and congestion monitoring.

---

## Overview

The system ingests traffic video and produces a fully annotated output with per-frame analytics. It is designed to run on CPU hardware using a lightweight YOLOv8 model while maintaining stable tracking and meaningful traffic insights.

**Capabilities:**

* Real-time vehicle detection and classification
* Persistent multi-object tracking across frames
* In/Out traffic flow counting via line-crossing analytics
* Per-category vehicle statistics (cars, trucks, buses, motorcycles)
* Relative speed estimation using frame-to-frame displacement
* Traffic density and congestion classification
* FPS monitoring and annotated output video generation

---

## Tech Stack

| Layer                | Technology           |
| -------------------- | -------------------- |
| Language             | Python               |
| Computer Vision      | OpenCV               |
| Detection            | YOLOv8 (Ultralytics) |
| Tracking & Analytics | Supervision          |

---

## System Pipeline

```text
Video Input
    -> YOLOv8 Vehicle Detection
    -> Multi-Object Tracking
    -> Analytics Layer (speed, density, flow)
    -> Visualization & Output Video Generation
```

---

## Architecture

### Detection

YOLOv8 Nano performs per-frame vehicle detection, balancing inference speed and accuracy for CPU based real-time processing.

### Tracking

Supervision's tracking utilities assign persistent IDs across frames, enabling temporal analysis of individual vehicles throughout their time in the scene.

### Analytics

Built on top of tracked detections, the analytics layer computes:

* Per frame and cumulative vehicle counts by class
* In/Out crossing counts via `LineZone` triggers
* Pixel-per-frame speed from positional displacement between frames
* Congestion state (LOW / MEDIUM / HIGH) based on active tracked vehicle density

### Visualization

All analytics are rendered as real-time overlays using OpenCV and Supervision annotators bounding boxes, tracking IDs, speed labels, flow counters, and congestion state and written to an output video file.

---

## Engineering Decisions

**YOLOv8 Nano** was chosen over larger variants to maintain real-time inference on CPU hardware without sacrificing tracking stability on highway scale scenes.

**Supervision** was used for line-crossing analytics, anchor extraction, and visualization, reducing boilerplate and keeping the analytics layer clean and composable.

**Pixel/frame speed estimation** was chosen over real-world speed calibration to keep the system hardware-agnostic and lightweight. Calibrated speed estimation is noted as a future improvement.

---

## Challenges

**Detection-Performance Tradeoff** — Inference speed and tracking stability had to be balanced carefully on CPU hardware, addressed through lightweight model selection and inference optimization.

**Tracking Stability** — Moving vehicles entering and exiting the frame required careful handling of tracker initialization and ID persistence.

**Line-Crossing Reliability** — Accurate counts depend on consistent tracking through the line zone. False triggers from unstable detections were a key consideration in line placement.

---

## Future Improvements

* Lane wise traffic analytics
* ROI based road masking
* Heatmap visualization for traffic density
* Interactive dashboard for live traffic monitoring

---

## Output

The pipeline generates and stores a processed output video containing all real-time analytics overlays and tracking visualizations.

Generated analytics include:

* Vehicle tracking visualization
* Flow counters
* Vehicle category analytics
* Relative speed labels
* Traffic density state
* FPS monitoring
