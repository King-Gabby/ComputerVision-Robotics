import cv2 as cv
import mediapipe as mp
import time
import math

#Utility 
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

#MediaPipe Setup 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

#Finger Config
FINGERS = {
    "Index": (8, 6),
    "Middle": (12, 10),
    "Ring": (16, 14),
    "Pinky": (20, 18)
}

PINCH_THRESHOLD = 40  # pixels (adjust based on camera distance)

#Camera 
cap = cv2.VideoCapture(0)
prev_time = 0

#Pinch & Drag State
pinch_active = False
drag_object_pos = (300, 300)
object_radius = 50

#Functions
def detect_fingers_up(landmarks):
    fingers = []
    for name, (tip, pip) in FINGERS.items():
        if landmarks[tip][1] < landmarks[pip][1]:
            fingers.append(name)
    return fingers

#Main Loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            landmarks = {}

            # Store landmarks
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks[idx] = (cx, cy)
                cv.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            #Pinch Detection
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            pinch_dist = distance(thumb_tip, index_tip)
            cv.line(frame, thumb_tip, index_tip, (255, 0, 255), 2)
            cv.putText(frame, f"Pinch Dist: {int(pinch_dist)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if pinch_dist < PINCH_THRESHOLD:
                if not pinch_active:
                    pinch_active = True
                    print("Click!")
                drag_object_pos = index_tip
                cv.putText(frame, "PINCH & DRAG", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            else:
                if pinch_active:
                    pinch_active = False
                    print("Release!")
                
                # Finger state detection
                fingers_up = detect_fingers_up(landmarks)
                y_offset = 120
                if fingers_up:
                    for finger in fingers_up:
                        cv.putText(frame, f"{finger} UP", (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        y_offset += 30
                else:
                    cv.putText(frame, "No fingers UP", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw draggable object
    cv2.circle(frame, drag_object_pos, object_radius, (0, 0, 255), -1)

    #FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("HandTracker Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()
