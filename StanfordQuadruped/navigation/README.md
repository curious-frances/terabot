# Pupper Navigation System

This module enables the Pupper robot to navigate towards detected objects using computer vision and YOLO object detection.

## Overview

The navigation system allows the Pupper to:
- Detect objects in its camera feed using a YOLO model
- Navigate towards a specified target object
- Maintain a safe distance from the target
- Center the target in its field of view
- Avoid obstacles while moving

## System Components

### 1. NavigationController
- Handles high-level navigation logic
- Calculates control commands based on object position
- Implements safety features and movement constraints
- Uses proportional control for smooth movements

### 2. YOLODetector
- Manages object detection using a TFLite YOLO model
- Processes camera frames for detection
- Provides detection results with bounding boxes and confidence scores
- Maps class IDs to human-readable names

### 3. PupperNavigation
- Main class that integrates all components
- Manages camera input and visualization
- Handles user input and command-line arguments
- Provides real-time feedback and status updates

## Prerequisites

- Python 3.7+
- OpenCV
- TensorFlow Lite
- Pupper robot hardware and software
- Camera connected to the robot



terabot/
├── StanfordQuadruped/
│   ├── src/
│   │   ├── navigation/
│   │   │   ├── __init__.py
│   │   │   ├── main_navigation.py
│   │   │   ├── navigation_controller.py
│   │   │   └── yolo_detector.py
│   │   ├── Controller.py
│   │   ├── State.py
│   │   └── ...
│   ├── pupper/
│   │   ├── Config.py
│   │   ├── Kinematics.py
│   │   └── ...
│   └── ...
└── model/
    └── best_int8.tflite