import time
import math
from collections import deque, Counter
import pyautogui
import cv2
from hand_tracker.tracker import extract_landmarks
from hand_tracker.gestures import detect_gesture

#PyAutoGUI Setup
pyautogui.FAILSAFE = True
SCREEN_W, SCREEN_H = pyautogui.size()

# Config 
SMOOTHING = 6
CLICK_DEBOUNCE = 0.4
DRAG_START_DISTANCE = 15

#State 
prev_x, prev_y = 0, 0
prev_scroll_y = None
pinch_active = False
pinch_start_pos = None
dragging = False
last_click_time = 0

#Camera 
cap = cv2.VideoCapture(0)
prev_time = time.time()

#Main Loop 
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    landmarks, frame = extract_landmarks(frame, draw=True)
    gesture = detect_gesture(landmarks) if landmarks else "NONE"

    now = time.time()

    if landmarks:
        index_x, index_y = landmarks[8]
        h, w, _ = frame.shape
        screen_x = int(index_x * SCREEN_W / w)
        screen_y = int(index_y * SCREEN_H / h)

        curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
        curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

        #Cursor Move 
        if gesture == "POINT" and not pinch_active:
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

        #Left Pinch Drag & Click 
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

        #Release 
        else:
            if dragging:
                pyautogui.mouseUp()
                dragging = False
                print("Drag End")

            if pinch_active and now - last_click_time > CLICK_DEBOUNCE:
                pyautogui.click()
                last_click_time = now
                print("Click")

            pinch_active = False
            pinch_start_pos = None

        #Right Click 
        if gesture == "PINCH_MIDDLE" and now - last_click_time > CLICK_DEBOUNCE:
            pyautogui.click(button="right")
            last_click_time = now
            print("Right Click")

        #Scroll 
        if gesture == "SCROLL":
            if prev_scroll_y is None:
                prev_scroll_y = index_y
            else:
                dy = prev_scroll_y - index_y
                if abs(dy) > 5:  # SCROLL_DEADZONE
                    pyautogui.scroll(int(dy * 1.2))  # SCROLL_SENSITIVITY
                prev_scroll_y = index_y
        else:
            prev_scroll_y = None

    #FPS 
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #Show Frame 
    cv2.putText(frame, f"Gesture: {gesture}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.imshow("HandMouse Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()
