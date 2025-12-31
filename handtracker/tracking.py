import cv2
import mediapipe as mp
import math

#Utility 
def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Core Functions 
def extract_landmarks(frame, draw=True):
    """
    Process a BGR frame and return hand landmarks as a dict {id: (x, y)}.
    
    Parameters:
        frame : np.array
            BGR image
        draw : bool
            Whether to draw landmarks on frame
    
    Returns:
        landmarks : dict or None
            Dictionary of landmark coordinates
        frame : np.array
            Frame with landmarks drawn if draw=True
    """
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if not result.multi_hand_landmarks:
        return None, frame

    hand_landmarks = result.multi_hand_landmarks[0]

    if draw:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    h, w, _ = frame.shape
    landmarks = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        landmarks[idx] = (int(lm.x * w), int(lm.y * h))

    return landmarks, frame
