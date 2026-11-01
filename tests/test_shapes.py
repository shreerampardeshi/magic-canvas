import cv2
import numpy as np

from shape_recognizer import detect_shapes


def blank():
    return np.zeros((400, 400, 3), np.uint8)


def test_triangle():
    img = blank()
    pts = np.array([[200, 60], [80, 300], [320, 300]], np.int32)
    cv2.polylines(img, [pts], True, (255, 255, 255), 4)
    assert detect_shapes(img)[0]["type"] == "triangle"


def test_circle():
    img = blank()
    cv2.circle(img, (200, 200), 100, (255, 255, 255), 4)
    assert detect_shapes(img)[0]["type"] == "circle"


def test_square():
    img = blank()
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), 4)
    assert detect_shapes(img)[0]["type"] == "square"
