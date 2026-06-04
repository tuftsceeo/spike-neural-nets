"""
main.py — Trains the model, builds the hub program, uploads and runs it.

Run this file from VS Code after:
  pip install torch numpy bleak
"""

import asyncio
from DirectionClassifier import train, extract_weights
from spike import SpikeHub

# ══════════════════════════════════════════════════════════════════════════════
# Arrow pixel patterns for the hub's 5×5 LED display.
# 25 values (0–9 brightness), row-major, top-left first.
# ══════════════════════════════════════════════════════════════════════════════

ARROWS = {
    # Up  ↑
    "up":    [0,0,9,0,0,
              0,9,9,9,0,
              9,0,9,0,9,
              0,0,9,0,0,
              0,0,9,0,0],
    # Right  →
    "right": [0,0,9,0,0,
              0,0,0,9,0,
              9,9,9,9,9,
              0,0,0,9,0,
              0,0,9,0,0],
    # Down  ↓
    "down":  [0,0,9,0,0,
              0,0,9,0,0,
              9,0,9,0,9,
              0,9,9,9,0,
              0,0,9,0,0],
    # Left  ←
    "left":  [0,0,9,0,0,
              0,9,0,0,0,
              9,9,9,9,9,
              0,9,0,0,0,
              0,0,9,0,0],
}


# ══════════════════════════════════════════════════════════════════════════════
# Hub program builder
# ══════════════════════════════════════════════════════════════════════════════

def build_hub_program(weights: dict) -> str:
    """
    Return a self-contained MicroPython string ready to run on the SPIKE hub.
    The trained weights are embedded as Python list literals — no dependencies
    needed on the hub side.
    """

    def fmt(lst):
        if isinstance(lst[0], list):
            rows = ",\n        ".join(str(row) for row in lst)
            return f"[\n        {rows}\n    ]"
        return str(lst)

    return f"""\
# direction_classifier.py — auto-generated, runs on SPIKE Prime hub
# Classifies motor angle → Up / Right / Down / Left and shows arrow on display.

import hub
import math
import time

# ── Trained weights (baked in from PC training) ───────────────────────────────
W1 = {fmt(weights['w1'])}
B1 = {fmt(weights['b1'])}
W2 = {fmt(weights['w2'])}
B2 = {fmt(weights['b2'])}
W3 = {fmt(weights['w3'])}
B3 = {fmt(weights['b3'])}

# ── Arrow pixel patterns for the 5×5 display ─────────────────────────────────
ARROWS = {{
    0: {ARROWS['up']},    # Up
    1: {ARROWS['right']}, # Right
    2: {ARROWS['down']},  # Down
    3: {ARROWS['left']},  # Left
}}

CLASS_NAMES = ["Up", "Right", "Down", "Left"]

# ── Pure-Python neural net forward pass ───────────────────────────────────────
def relu(x):
    return [v if v > 0 else 0.0 for v in x]

def linear(x, w, b):
    return [
        sum(w[i][j] * x[j] for j in range(len(x))) + b[i]
        for i in range(len(b))
    ]

def predict(angle_deg):
    rad = math.radians(angle_deg)
    x   = [math.sin(rad), math.cos(rad)]
    x   = relu(linear(x, W1, B1))
    x   = relu(linear(x, W2, B2))
    x   = linear(x, W3, B3)
    return x.index(max(x))

# ── Display helper ────────────────────────────────────────────────────────────
def show_arrow(direction_idx):
    pixels = ARROWS[direction_idx]
    hub.display.show(hub.Image(pixels))

# ── Main loop ─────────────────────────────────────────────────────────────────
# Motor should be plugged into port A.
print("Direction classifier running. Press hub button to stop.")

motor = hub.port.A.motor
last_direction = None

while True:
    angle = motor.get()[1]       # absolute position: -180 to 179
    angle = (angle + 360) % 360  # normalise to 0-359

    direction = predict(angle)

    if direction != last_direction:
        show_arrow(direction)
        print(CLASS_NAMES[direction], angle)
        last_direction = direction

    time.sleep_ms(100)
"""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    # Step 1: train the model on your PC
    model   = train()
    weights = extract_weights(model)

    # Step 2: build the self-contained MicroPython program with weights baked in
    program_source = build_hub_program(weights)

    # (Optional) save a local copy so you can inspect what gets uploaded
    with open("hub_program.py", "w", encoding="utf-8") as f:
        f.write(program_source)
    print("Hub program written to hub_program.py")

    # Step 3: connect to the hub, upload, and start
    async with SpikeHub() as hub:
        await hub.upload_program(program_source, slot=0,
                                 filename="direction_classifier.py")
        #await hub.start_program(slot=0)

        # Stream console output for 60 seconds so you can see predictions
        #await hub.stream_console(duration=60.0)


if __name__ == "__main__":
    asyncio.run(main())