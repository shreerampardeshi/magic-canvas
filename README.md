# Magic Canvas

An AI-powered math and physics solver. Draw equations and diagrams in the air with your index finger; Magic Canvas recognises the shapes with computer vision and sends the drawing plus those hints to an LLM that produces a step-by-step solution in real time.

## How it works

```
Webcam -> MediaPipe hand tracking -> gesture state machine -> virtual canvas
                                                                   |
                                    OpenCV contour analysis (shapes, angles, sides)
                                                                   |
                          canvas image + shape hints -> LLM reasoning -> worked solution
```

| Module | Role |
|---|---|
| `hand_tracker.py` | MediaPipe landmarks and gesture classification |
| `shape_recognizer.py` | Contour-based detection of lines, triangles, rectangles, circles and polygons, with angles and side lengths |
| `solver.py` | Sends the cropped drawing and CV hints to the LLM |
| `app.py` | Real-time loop, drawing, UI windows |

## Gestures

| Gesture | Action |
|---|---|
| Index finger up | Draw |
| Index + middle up | Move without drawing |
| Open palm (hold) | Clear canvas |
| Fist (hold) | Solve |

Keyboard fallback: `s` solve, `c` clear, `q` quit.

## Setup

```bash
git clone https://github.com/shreerampardeshi/magic-canvas.git
cd magic-canvas
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # add your ANTHROPIC_API_KEY
python app.py
```

Run the tests with `pip install pytest && pytest`.

## Limitations and ideas

- Pixel lengths are not real units, so the LLM relies on handwritten labels for numbers.
- Planned: stroke undo, multi-hand support, offline handwriting OCR, saving solved problems.

## Tech stack

Python, OpenCV, MediaPipe, NumPy, Anthropic API.
