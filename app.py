"""Magic Canvas: draw in the air, get a worked solution.

Gestures:  index finger = draw | index+middle = move | open palm = clear | fist (hold) = solve
Keys:      s = solve, c = clear, q = quit
"""
import textwrap
import threading

import cv2
import numpy as np

from hand_tracker import HandTracker
from shape_recognizer import describe, detect_shapes
from solver import Solver

WIDTH, HEIGHT = 960, 540
HOLD_FRAMES = 12  # frames a gesture must be held before it triggers
BRUSH = 6


def crop_for_llm(canvas):
    """White-on-black canvas -> black-on-white crop with padding."""
    ys, xs = np.where(canvas.max(axis=2) > 0)
    if len(xs) == 0:
        return None
    pad = 30
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, canvas.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, canvas.shape[0])
    return 255 - canvas[y0:y1, x0:x1]


def render_panel(text, size=(520, 540)):
    panel = np.full((size[1], size[0], 3), 255, np.uint8)
    y = 30
    for para in text.split("\n"):
        for line in textwrap.wrap(para, 52) or [""]:
            cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
            y += 22
            if y > size[1] - 10:
                return panel
    return panel


class MagicCanvas:
    def __init__(self):
        self.tracker = HandTracker()
        self.solver = Solver()
        self.canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        self.prev = None
        self.hold = {"clear": 0, "solve": 0}
        self.result = "Draw a problem with your index finger.\nMake a fist to solve."
        self.busy = False

    def clear(self):
        self.canvas[:] = 0
        self.prev = None
        self.result = "Canvas cleared."

    def solve_async(self):
        if self.busy:
            return
        crop = crop_for_llm(self.canvas)
        if crop is None:
            self.result = "Nothing drawn yet."
            return
        hints = describe(detect_shapes(self.canvas))
        self.busy = True
        self.result = "Solving..."

        def work():
            try:
                self.result = self.solver.solve(crop, hints)
            except Exception as exc:  # network / API errors should not crash the UI
                self.result = f"Error: {exc}"
            finally:
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def update(self, gesture, tip):
        if gesture == "draw" and tip:
            if self.prev:
                cv2.line(self.canvas, self.prev, tip, (255, 255, 255), BRUSH)
            self.prev = tip
        else:
            self.prev = tip if gesture == "move" else None
        for g in self.hold:
            self.hold[g] = self.hold[g] + 1 if gesture == g else 0
        if self.hold["clear"] == HOLD_FRAMES:
            self.clear()
        if self.hold["solve"] == HOLD_FRAMES:
            self.solve_async()

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(cv2.flip(frame, 1), (WIDTH, HEIGHT))
            frame = self.tracker.process(frame)
            self.update(self.tracker.gesture(), self.tracker.index_tip)
            mask = self.canvas.max(axis=2) > 0
            frame[mask] = (0, 255, 255)
            cv2.putText(frame, f"gesture: {self.tracker.gesture()}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Magic Canvas", frame)
            cv2.imshow("Solution", render_panel(self.result))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                self.clear()
            if key == ord("s"):
                self.solve_async()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MagicCanvas().run()
