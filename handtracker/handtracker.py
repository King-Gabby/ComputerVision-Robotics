import cv2 as cv
import mediapipe as mp
import time
import math


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

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
    "Pinky": (20, 18) #i use this to signify the finger tips and joints
}

PINCH_THRESHOLD = 40  #pixels i can change depending on the cam distance


# Camera
cap = cv.VideoCapture(0)
prev_time = 0

#pinch n drag
pinch_active = False
drag_object_pos = (300, 300)  
object_radius = 50


def detect_fingers_up(landmarks):
    fingers = []
    for name, (tip, pip) in FINGERS.items():
        if landmarks[tip][1] < landmarks[pip][1]:
            fingers.append(name)
    return fingers 


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            landmarks = {}

            # Store landmarks
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks[id] = (cx, cy)
                cv.circle(frame, (cx, cy), 3, (0, 255, 0), -1)



            # Pinch detection
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            pinch_dist = distance(thumb_tip, index_tip)
            cv.line(frame, thumb_tip, index_tip, (255, 0, 255), 2)
            cv.putText(frame, f"Pinch Dist: {int(pinch_dist)}", (10, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if pinch_dist < PINCH_THRESHOLD:
                #pinch active
                if not pinch_active:
                    #starting
                    pinch_active = True
                    print("Click!")
                drag_object_pos = index_tip
                cv.putText(frame, "PINCH & DRAG", (10, 120),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            else:
                #releasing pinch
                if pinch_active:
                    pinch_active = False
                    print("Release!")
                #Finger state mode
                fingers_up = []
                for finger_name, (tip_id, pip_id) in FINGERS.items():
                    if landmarks[tip_id][1] < landmarks[pip_id][1]:
                        fingers_up.append(finger_name)

                y_offset = 120
                if fingers_up:
                    for finger in fingers_up:
                        cv.putText(frame, f"{finger} UP", (10, y_offset),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        y_offset += 30
                else:
                    cv.putText(frame, "No fingers UP", (10, 120),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    #Draw draggable object
    cv.circle(frame, drag_object_pos, object_radius, (0, 0, 255), -1)

    #fps
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if curr_time != prev_time else 0
    prev_time = curr_time
    cv.putText(frame, f"FPS: {fps}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow("Hand Gesture Click & Drag", frame)

    if cv.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
cv.destroyAllWindows()
