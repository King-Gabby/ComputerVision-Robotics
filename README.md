# AI & Computer Vision Engineer | Machine Learning | Deep Learning | Robotics

# ComputerVision-Robotics

A collection of computer vision projects for **hand gesture recognition and control**.

This repository features two main projects:

1. Hand Tracker – detects and visualizes hand landmarks and gestures.
2. Hand Mouse Controller – uses hand gestures to control the OS mouse cursor.



## Hand Tracker

**Purpose:** Detect hand landmarks, identify which fingers are up, and recognize gestures.

### Features
- Real-time hand landmark detection using **MediaPipe** and **OpenCV**
- Gesture recognition (PINCH, POINT, SCROLL)
- Frame smoothing to reduce flicker
- Both full demo and modular engine

### Files
- `full_handtracker_demo.py` – all-in-one demo
- `tracker.py` – MediaPipe hand tracker (landmarks)
- `gestures.py` – gesture detection engine
- `demo.py` – modular demo combining tracker + gestures


### How to RUN
-- `Hand Tracker full demo` -- cd hand_tracker
                              python full_handtracker_demo.py

-- `Hand Tracker modular demo` -- cd hand_tracker
                                   python demo.py
-- `Hand Mouse Controller` -- cd hand_mouse_controller
                              python controller.py
  

## Note:
Make sure Python 3.9+ is installed

Install dependencies:
pip install -r requirements.txt
Press k to exit the demo windows



### About Me:

AI & Computer Vision Engineer specializing in Machine Learning, Deep Learning, and intelligent system design.

I build scalable AI systems that transform visual and structured data into actionable intelligence. My work focuses on real-world deployment, from model development and optimization to production-ready architectures.

# Core interests:

Computer Vision & Robotic Perception

Applied Machine Learning

Intelligent Automation Systems

Real-time Inference & Optimization

AI System Architecture

I approach engineering with precision, clarity, and long-term scalability in mind.



### What I Build:

I design and develop:

Computer Vision pipelines for object detection, tracking, and pose estimation

Deep Learning models for perception and decision-making systems

Real-time AI inference systems optimized for performance

Robotics-integrated vision systems

Intelligent applications powered by LLMs and multimodal AI

End-to-end machine learning workflows (data → model → deployment)

I prioritize:

Clean architecture

Reproducibility

Performance efficiency

Production-grade standards


### Tech Stack:

Languages

Python

C++

SQL

AI & Machine Learning

PyTorch

TensorFlow

Scikit-learn

OpenCV

Hugging Face

Robotics & Vision

ROS

Camera calibration & sensor fusion

Real-time perception systems

Deployment & Infrastructure

FastAPI

Docker

Linux

Git

Cloud deployment workflows



### Usage (Modular)
```python
from hand_tracker.tracker import extract_landmarks
from hand_tracker.gestures import detect_gesture

# Capture frame from camera
landmarks, frame = extract_landmarks(frame)
gesture = detect_gesture(landmarks)



