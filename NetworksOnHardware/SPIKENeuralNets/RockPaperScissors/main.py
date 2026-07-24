"""
main.py — Trains the model, builds the hub program, uploads and runs it.

Run this file from VS Code after:
  pip install torch numpy bleak
"""

import asyncio
from RPSClassifier import train, extract_weights
import os

# ══════════════════════════════════════════════════════════════════════════════
# Arrow pixel patterns for the hub's 5×5 LED display.
# 25 values (0–9 brightness), row-major, top-left first.
# ══════════════════════════════════════════════════════════════════════════════

LETTERS = {
    0: [1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 1, 0, 0,
        1, 0, 0, 1, 0],
    1: [1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0],
    2: [1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        1, 1, 1, 1, 1],
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
# rps_hub_program.py — auto-generated, runs on SPIKE Prime hub
# Classifies "hand" position as Rock, Paper, or Scissors.

import hub
from hub import light_matrix
import motor
from hub import port
import math
import time

# ── Trained weights (baked in from PC training) ───────────────────────────────
W1 = {fmt(weights['w1'])}
B1 = {fmt(weights['b1'])}
W2 = {fmt(weights['w2'])}
B2 = {fmt(weights['b2'])}
W3 = {fmt(weights['w3'])}
B3 = {fmt(weights['b3'])}

# Letter pixel patterns
LETTERS = {{
    0: {LETTERS[0]},
    1: {LETTERS[1]}, 
    2: {LETTERS[2]}, 
}}

CLASS_NAMES = ["Rock", "Paper", "Scissors"]

# neural net forward pass
def relu(x):
    return [v if v > 0 else 0.0 for v in x]

def linear(x, w, b):
    return [
        sum(w[i][j] * x[j] for j in range(len(x))) + b[i]
        for i in range(len(b))
    ]

def predict(fingers):
    finger_rads = [math.radians(fingers[0]), math.radians(fingers[1]), math.radians(fingers[2])]
    x = [math.sin(finger_rads[0]), math.cos(finger_rads[0]), math.sin(finger_rads[1]), math.cos(finger_rads[1]), math.sin(finger_rads[2]), math.cos(finger_rads[2])]
    x = relu(linear(x, W1, B1))
    x = relu(linear(x, W2, B2))
    x = linear(x, W3, B3)
    return x.index(max(x))

def show_letter(position_idx):
    pixels = LETTERS[position_idx]
    hub.light_matrix.show(pixels)

# Main Loop
print("Classifier running")

last_position = None

while True:
    fingers = [motor.absolute_position(port.B), motor.absolute_position(port.D), motor.absolute_position(port.F)]    # absolute position: -180 to 179
    fingers = [(fingers[0] + 360) % 360, (fingers[1] + 360) % 360, (fingers[2] + 360) % 360] # normalise to 0-359

    position = predict(fingers)

    if position != last_position:
        show_letter(position)
        print(CLASS_NAMES[position])
        last_position = position

    time.sleep_ms(100)
"""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    # Train model
    model   = train()
    weights = extract_weights(model)

    # Build SPIKE program
    program_source = build_hub_program(weights)

    # Save a local copy
    # Get the absolute path of the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Safely join the directory path and the filename together
    file_path = os.path.join(script_dir, "rps_hub_program.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(program_source)
    print("Hub program written to rps_hub_program.py")


if __name__ == "__main__":
    asyncio.run(main())