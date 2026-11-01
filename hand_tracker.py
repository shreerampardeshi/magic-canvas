"""Hand tracking and gesture detection built on MediaPipe Hands."""
import cv2
import mediapipe as mp

INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


class HandTracker:
    def __init__(self, detection_conf=0.7, tracking_conf=0.6):
        self._mp_hands = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarks = None  # list of (x, y) pixel coordinates

    def process(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        self.landmarks = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            self.landmarks = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]
            if draw:
                self._drawer.draw_landmarks(frame, hand, self._mp_hands.HAND_CONNECTIONS)
        return frame

    def fingers_up(self):
        """Return [index, middle, ring, pinky] booleans (thumb is ignored)."""
        if not self.landmarks:
            return None
        lm = self.landmarks
        pairs = [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                 (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]
        return [lm[tip][1] < lm[pip][1] for tip, pip in pairs]

    def gesture(self):
        """Map finger state to a named gesture."""
        up = self.fingers_up()
        if up is None:
            return "none"
        if up == [True, False, False, False]:
            return "draw"      # index finger only
        if up == [True, True, False, False]:
            return "move"      # index + middle: move without drawing
        if all(up):
            return "clear"     # open palm
        if not any(up):
            return "solve"     # closed fist
        return "none"

    @property
    def index_tip(self):
        return self.landmarks[INDEX_TIP] if self.landmarks else None
