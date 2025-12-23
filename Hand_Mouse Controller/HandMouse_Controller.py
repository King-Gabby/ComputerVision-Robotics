import cv2 as cv
import mediapipe as mp
import time
import math
import pyautogui
from collections import deque, Counter
import csv

pyautogui.FAILSAFE = True

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

FINGERS = {
    "Index": (8, 6),
    "Middle": (12, 10),
    "Ring": (16, 14),
    "Pinky": (20, 18)
}

# Config
PINCH_THRESHOLD = 40
SCROLL_SPEED = 30
SMOOTHING = 6
GESTURE_WINDOW = 7
prev_scroll_y = None
SCROLL_SENSITIVITY = 1.5
SCROLL_DEADZONE = 5

SCREEN_W, SCREEN_H = pyautogui.size()
prev_x, prev_y = 0, 0
mouse_down = False

gesture_history = deque(maxlen=GESTURE_WINDOW)
calibration_mode = False

# Gesture logging
log_file = open("gesture_log.csv", "w", newline="")
logger = csv.writer(log_file)
logger.writerow(["timestamp", "gesture"])

def smooth_gesture(g):
    gesture_history.append(g)
    return Counter(gesture_history).most_common(1)[0][0]

def fingers_up(landmarks):
    up = []
    for name, (tip, pip) in FINGERS.items():
        if landmarks[tip][1] < landmarks[pip][1]:
            up.append(name)
    return up

def detect_gesture(landmarks):
    thumb = landmarks[4]
    index = landmarks[8]
    middle = landmarks[12]

    pinch_index = distance(thumb, index)
    pinch_middle = distance(thumb, middle)

    up = fingers_up(landmarks)

    if pinch_index < PINCH_THRESHOLD:
        return "PINCH"
    elif pinch_middle < PINCH_THRESHOLD:
        return "RIGHT_CLICK"
    elif up == ["Index", "Middle"]:
        return "SCROLL"
    elif up == ["Index"]:
        return "POINT"
    else:
        return "NONE"

cap = cv.VideoCapture(0)
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]  # Primary hand only
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        landmarks = {}

        for id, lm in enumerate(hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks[id] = (cx, cy)

        raw_gesture = detect_gesture(landmarks)
        gesture = smooth_gesture(raw_gesture)

        logger.writerow([time.time(), gesture])

        index_x, index_y = landmarks[8]
        screen_x = int(index_x * SCREEN_W / w)
        screen_y = int(index_y * SCREEN_H / h)

        curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
        curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

        # ACTION LAYER
        if gesture == "POINT":
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

        elif gesture == "PINCH":
            if not mouse_down:
                pyautogui.mouseDown()
                mouse_down = True
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

        else:
            if mouse_down:
                pyautogui.mouseUp()
                mouse_down = False

        if gesture == "RIGHT_CLICK":
            pyautogui.click(button="right")
            time.sleep(0.3)  # debounce

        if gesture == "SCROLL":
            if prev_scroll_y is None:
                prev_scroll_y = index_y
            else:
                dy = prev_scroll_y - index_y  # inverted (natural scroll)

                if abs(dy) > SCROLL_DEADZONE:
                    scroll_amount = int(dy * SCROLL_SENSITIVITY)
                    pyautogui.scroll(scroll_amount)

                prev_scroll_y = index_y

            cv.putText(frame, "SCROLLING", (10, 160),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        else:
            prev_scroll_y = None


        cv.putText(frame, f"Gesture: {gesture}", (10, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if curr_time != prev_time else 0
    prev_time = curr_time

    cv.putText(frame, f"FPS: {fps}", (10, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow("Hand OS Controller", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('k'):
        break
    if key == ord('c'):
        calibration_mode = not calibration_mode
        print("Calibration toggled")

cap.release()
log_file.close()
cv.destroyAllWindows()
