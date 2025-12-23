import cv2 as cv
import mediapipe as mp
import time
import math
import pyautogui

pyautogui.FAILSAFE = True

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
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

PINCH_THRESHOLD = 40

# Screen
SCREEN_W, SCREEN_H = pyautogui.size()

# Cursor smoothing
prev_x, prev_y = 0, 0
SMOOTHING = 5

# Mouse state
mouse_down = False

def detect_fingers_up(landmarks):
    fingers = []
    for name, (tip, pip) in FINGERS.items():
        if landmarks[tip][1] < landmarks[pip][1]:
            fingers.append(name)
    return fingers

def detect_gesture(landmarks):
    pinch_dist = distance(landmarks[4], landmarks[8])
    fingers = detect_fingers_up(landmarks)

    if pinch_dist < PINCH_THRESHOLD:
        return "PINCH", pinch_dist
    elif fingers == ["Index"]:
        return "POINT", pinch_dist
    else:
        return "NONE", pinch_dist


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
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            landmarks = {}

            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks[id] = (cx, cy)

            gesture, pinch_dist = detect_gesture(landmarks)

            # MOUSE CONTROL
            index_x, index_y = landmarks[8]

            # Map camera . screen
            screen_x = int(index_x * SCREEN_W / w)
            screen_y = int(index_y * SCREEN_H / h)

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
            curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

            if gesture == "POINT":
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                cv.putText(frame, "MOVE", (10, 120),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            elif gesture == "PINCH":
                if not mouse_down:
                    pyautogui.mouseDown()
                    mouse_down = True
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                cv.putText(frame, "DRAG", (10, 120),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            else:
                if mouse_down:
                    pyautogui.mouseUp()
                    mouse_down = False

            cv.putText(frame, f"Gesture: {gesture}", (10, 80),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    
    # FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if curr_time != prev_time else 0
    prev_time = curr_time

    cv.putText(frame, f"FPS: {fps}", (10, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow("Hand Mouse Controller", frame)
    if cv.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
cv.destroyAllWindows()
