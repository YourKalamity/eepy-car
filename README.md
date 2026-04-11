# eepy-car

> Real-time driver drowsiness and distraction detection system.

Built as part of my Final Year Project for BSc Computer Science @ Aston University.

[![Tests](https://github.com/YourKalamity/eepy-car/actions/workflows/test.yml/badge.svg)](https://github.com/YourKalamity/eepy-car/actions/workflows/test.yml)

---

## Overview

eepy-car is a lightweight, real-time driver monitoring system that detects both **drowsiness** and **distraction** from a single forward-facing camera

**Drowsiness** is detected by computing the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) from MediaPipe facial landmarks.
**Distraction** is detected using a novel approach: an AprilTag fiducial marker affixed to the vehicle headrest acts as a spatial ground truth for the forward-facing direction. The angular offset between the driver's head pose and the tag's orientation is decomposed into yaw (left/right) and pitch (up/down) components.

Both detection branches run concurrently and feed into a multi-indicator decision model. 
Each indicator accumulates a time-weighted score that grows proportionally to the magnitude and duration of the deviation. 
A weighted fusion step combines the scores into composite drowsiness and distraction scores, which are classified into five alert levels.

---

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- A camera (tested with iPhone 15 Pro Max via Apple Continuity Camera)
- An AprilTag (tag36h11 family, ID 250) affixed to the vehicle headrest
- Camera calibration data (see [Calibration](#calibration))

---

## Installation

```bash
git clone https://github.com/YourKalamity/eepy-car.git
cd eepy-car
uv sync
```

---

## Calibration

Camera calibration is required for accurate AprilTag pose estimation. Print a 9×6 chessboard pattern, affix it to a rigid flat surface, then run:

```bash
uv run python experiments/calibration.py --square 0.025 --samples 20
```

Hold the board at varied distances and angles, pressing **Space** to capture each sample. The calibration file is saved to `calibration_output/calibration.npz`.

---

## AprilTag Setup

Print or 3D-print an AprilTag from the `tag36h11` family with ID `250`. Affix it to the centre of the vehicle headrest facing the camera.

The tag only needs to be visible at system startup. Once the pose is acquired, the driver's head can occlude it — the last known pose remains valid for the entire session.

> **Tip:** 3D-printing in matte PLA produces a rigid, non-reflective tag that detects significantly more reliably than paper prints.

---

## Configuration

All thresholds, weights, and output settings are controlled via `config.json` 

## Usage

```bash
uv run python src/eepy_car/main.py
```

On startup:
1. The system checks for required files and initialises all components
2. Point the camera at the headrest tag
4. The system begins monitoring the state

Press **Q** or **ESC** to quit.

---

## Running Tests

```bash
uv run pytest tests/ -v
uv run coverage run -m pytest tests/ && uv run coverage report -m
```

---

## Alert Levels

| Level | Trigger |
|---|---|
| `DROWSINESS_WARNING` | Drowsiness score exceeds 0.4 |
| `CRITICAL_DROWSINESS` | Drowsiness score exceeds 0.75 |
| `DISTRACTION_WARNING` | Distraction score exceeds 7.0 |
| `CRITICAL_DISTRACTION` | Distraction score exceeds 20.0 |

Drowsiness is always evaluated before distraction. 
All scores decay when indicators return to normal.

---

## Academic Context

This system was developed as part of my Final Year Project for BSc Computer Science at Aston University (2025–2026). 
The novel contribution is the use of an AprilTag fiducial marker on the vehicle headrest as a calibration-free spatial ground truth for gaze zone estimation, something not previously explored in the driver monitoring literature.

The drowsiness component is evaluated against the [YawDD dataset](https://ieee-dataport.org/documents/yawdd-yawning-detection-dataset). The distraction component is evaluated using a customtest setup.

---
