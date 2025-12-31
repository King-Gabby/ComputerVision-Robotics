import cv2
import time
from hand_tracker.tracker import extract_landmarks
from hand_tracker.gestures import detect_gesture

#Camera Setup 
cap = cv2.VideoCapture(0)
prev_time = 0

#Pinch & Drag State (for visualization)
pinch_active = False
drag_object_pos = (300, 300)
object_radius = 50

#Main Loop 
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    #Landmark Extraction
    landmarks, frame = extract_landmarks(frame, draw=True)

    gesture = "NONE"
    if landmarks:
        gesture = detect_gesture(landmarks)

        #Pinch Simulation
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist = ((thumb_tip[0]-index_tip[0])**2 + (thumb_tip[1]-index_tip[1])**2)**0.5

        if pinch_dist < 40:  # PINCH_THRESHOLD
            pinch_active = True
            drag_object_pos = index_tip
            cv2.putText(frame, "PINCH & DRAG", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        else:
            pinch_active = False

        #Finger State Display 
        y_offset = 120
        from hand_tracker.gestures import fingers_up
        fingers = fingers_up(landmarks)
        if fingers:
            for finger in fingers:
                cv2.putText(frame, f"{finger} UP", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30
        else:
            cv2.putText(frame, "No fingers UP", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    #Draw Draggable Object 
    cv2.circle(frame, drag_object_pos, object_radius, (0, 0, 255), -1)

    #FPS 
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-6))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #Show Frame
    cv2.imshow("HandTracker Modular Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()
