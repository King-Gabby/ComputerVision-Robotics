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



### Usage (Modular)
```python
from hand_tracker.tracker import extract_landmarks
from hand_tracker.gestures import detect_gesture

# Capture frame from camera
landmarks, frame = extract_landmarks(frame)
gesture = detect_gesture(landmarks)



