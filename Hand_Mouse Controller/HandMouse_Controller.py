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

#mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

#finger config
FINGERS = {
    "Index": (8, 6),
    "Middle": (12, 10),
    "Ring": (16, 14),
    "Pinky": (20, 18)
}

PINCH_THRESHOLD = 40
SMOOTHING = 6
SCROLL_DEADZONE = 5
SCROLL_SENSITIVITY = 1.2

CLICK_DEBOUNCE = 0.4
DRAG_START_DISTANCE = 15
GESTURE_WINDOW = 7

SCREEN_W, SCREEN_H = pyautogui.size()

prev_x, prev_y = 0, 0
prev_scroll_y = None

gesture_history = deque(maxlen=GESTURE_WINDOW)

pinch_active = False
pinch_start_pos = None
dragging = False
last_click_time = 0

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
        return "PINCH_INDEX"
    elif pinch_middle < PINCH_THRESHOLD:
        return "PINCH_MIDDLE"
    elif up == ["Index", "Middle"]:
        return "SCROLL"
    elif up == ["Index"]:
        return "POINT"
    else:
        return "NONE"

#cam
cap = cv.VideoCapture(0)
prev_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        landmarks = {}

        for i, lm in enumerate(hand.landmark):
            landmarks[i] = (int(lm.x * w), int(lm.y * h))

        raw_gesture = detect_gesture(landmarks)
        gesture = smooth_gesture(raw_gesture)
        logger.writerow([time.time(), gesture])

        index_x, index_y = landmarks[8]
        screen_x = int(index_x * SCREEN_W / w)
        screen_y = int(index_y * SCREEN_H / h)

        curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
        curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

        now = time.time()


        #cursor move
        if gesture == "POINT" and not pinch_active:
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

        #the left pinch drag and click
        elif gesture == "PINCH_INDEX":
            if not pinch_active:
                pinch_active = True
                pinch_start_pos = (curr_x, curr_y)
            else:
                dx = curr_x - pinch_start_pos[0]
                dy = curr_y - pinch_start_pos[1]
                move_dist = math.hypot(dx, dy)

                if move_dist > DRAG_START_DISTANCE:
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                        print("Drag Start")

                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

        #release
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False
                print("Drag End")

            if pinch_active:
                if now - last_click_time > CLICK_DEBOUNCE:
                    pyautogui.click()
                    last_click_time = now
                    print("Click")

            pinch_active = False
            pinch_start_pos = None

        #right click
        if gesture == "PINCH_MIDDLE":
            if now - last_click_time > CLICK_DEBOUNCE:
                pyautogui.click(button="right")
                last_click_time = now
                print("Right Click")

        #scroll
        if gesture == "SCROLL":
            if prev_scroll_y is None:
                prev_scroll_y = index_y
            else:
                dy = prev_scroll_y - index_y
                if abs(dy) > SCROLL_DEADZONE:
                    pyautogui.scroll(int(dy * SCROLL_SENSITIVITY))
                prev_scroll_y = index_y
        else:
            prev_scroll_y = None

        #ui
        cv.putText(frame, f"Gesture: {gesture}", (10, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    #fps
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if curr_time != prev_time else 0
    prev_time = curr_time

    cv.putText(frame, f"FPS: {fps}", (10, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow("Hand OS Controller by Gabriel", frame)

    if cv.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
log_file.close()
cv.destroyAllWindows()
