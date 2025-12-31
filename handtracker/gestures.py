from collections import deque, Counter
from .tracker import distance

#Configuration
FINGERS = {
    "Index": (8, 6),
    "Middle": (12, 10),
    "Ring": (16, 14),
    "Pinky": (20, 18)
}

PINCH_THRESHOLD = 40  # pixels
GESTURE_SMOOTHING = 7  # number of frames to smooth

# State
gesture_history = deque(maxlen=GESTURE_SMOOTHING)

#Functions 
def fingers_up(landmarks):
    """
    Returns a list of fingers that are up.
    landmarks: dict {id: (x, y)}
    """
    up = []
    for name, (tip, pip) in FINGERS.items():
        if landmarks[tip][1] < landmarks[pip][1]:
            up.append(name)
    return up

def detect_gesture(landmarks):
    """
    Classify gesture based on landmarks.
    Returns one of:
        "PINCH", "POINT", "SCROLL", "NONE"
    """
    if landmarks is None:
        return "NONE"

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]

    pinch_index = distance(thumb_tip, index_tip)
    pinch_middle = distance(thumb_tip, middle_tip)

    up = fingers_up(landmarks)

    #Gesture Logic
    if pinch_index < PINCH_THRESHOLD:
        gesture = "PINCH_INDEX"
    elif pinch_middle < PINCH_THRESHOLD:
        gesture = "PINCH_MIDDLE"
    elif up == ["Index", "Middle"]:
        gesture = "SCROLL"
    elif up == ["Index"]:
        gesture = "POINT"
    else:
        gesture = "NONE"

    # Smoothing
    gesture_history.append(gesture)
    smoothed_gesture = Counter(gesture_history).most_common(1)[0][0]

    return smoothed_gesture
