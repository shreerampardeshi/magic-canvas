"""Classical computer-vision shape recognition for hand-drawn diagrams.

The output is a compact textual description that is passed to the LLM as
structured hints alongside the raw image.
"""
import math

import cv2
import numpy as np


def _side_lengths(pts):
    return [float(np.linalg.norm(pts[i] - pts[(i + 1) % len(pts)])) for i in range(len(pts))]


def _angles(pts):
    out = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        out.append(float(math.degrees(math.acos(np.clip(cos, -1, 1)))))
    return out


def classify_contour(contour):
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    if min(rw, rh) / (max(rw, rh) + 1e-9) < 0.15:
        return "line", None
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    pts = approx.reshape(-1, 2).astype(float)
    n = len(pts)
    circularity = 4 * math.pi * area / (peri ** 2 + 1e-9)
    if circularity > 0.8 and n > 6:
        return "circle", None
    if n == 3:
        return "triangle", pts
    if n == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return ("square" if 0.9 <= w / h <= 1.1 else "rectangle"), pts
    return f"polygon({n} sides)", pts


def detect_shapes(canvas_bgr, min_extent=25):
    """Detect shapes on a canvas with white strokes on a black background."""
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) < min_extent:
            continue  # tiny blobs are handled by the LLM (digits, dots)
        kind, pts = classify_contour(c)
        info = {"type": kind, "bbox": (x, y, w, h)}
        if pts is not None:
            info["side_lengths_px"] = [round(s, 1) for s in _side_lengths(pts)]
            info["angles_deg"] = [round(a, 1) for a in _angles(pts)]
        shapes.append(info)
    return sorted(shapes, key=lambda s: (s["bbox"][1], s["bbox"][0]))


def describe(shapes):
    if not shapes:
        return "No closed geometric shapes detected."
    lines = []
    for i, s in enumerate(shapes, 1):
        text = f"{i}. {s['type']} at bbox={s['bbox']}"
        if "angles_deg" in s:
            text += f", angles={s['angles_deg']}, sides_px={s['side_lengths_px']}"
        lines.append(text)
    return "\n".join(lines)
