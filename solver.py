"""LLM reasoning layer: turns the canvas image + CV hints into a worked solution."""
import base64
import os

import cv2
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a math and physics tutor. The user drew a problem by hand: it may contain "
    "equations, handwritten numbers, geometric figures, or physics diagrams (forces, "
    "inclined planes, circuits, projectiles). First transcribe what you see in one line. "
    "Then solve it step by step with short steps, using the shape hints only as supporting "
    "evidence (pixel lengths are NOT real units unless labelled). If the drawing is "
    "ambiguous, state your assumption. End with a line starting 'Answer:'. "
    "Keep the whole response under 180 words, plain text, no markdown."
)


def _encode(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Could not encode canvas image")
    return base64.standard_b64encode(buf.tobytes()).decode()


class Solver:
    def __init__(self, model=None):
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model or os.getenv("MAGIC_CANVAS_MODEL", "claude-sonnet-5-5")

    def solve(self, canvas_img, shape_summary):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": _encode(canvas_img)}},
                    {"type": "text",
                     "text": f"Shape detector hints:\n{shape_summary}\n\nSolve the drawn problem."},
                ],
            }],
        )
        return "".join(b.text for b in message.content if b.type == "text").strip()
