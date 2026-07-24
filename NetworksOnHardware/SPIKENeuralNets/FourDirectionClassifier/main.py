"""
main.py — Trains the model, builds the hub program, uploads and runs it.

Run this file from VS Code after:
  pip install torch numpy bleak
"""

import asyncio
from DirectionClassifier import train, extract_weights
from NeuralNetOnSPIKE.Hubs.spike import SpikeHub
import os

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

# ── Arrow pixel patterns for the 5×5 display ─────────────────────────────────
ARROWS = {{
    0: light_matrix.IMAGE_ARROW_N,
    1: light_matrix.IMAGE_ARROW_E,
    2: light_matrix.IMAGE_ARROW_S,
    3: light_matrix.IMAGE_ARROW_W,
}}

CLASS_NAMES = ["Up", "Right", "Down", "Left"]

# neural net forward pass
def relu(x):
    return [v if v > 0 else 0.0 for v in x]

def linear(x, w, b):
    return [
        sum(w[i][j] * x[j] for j in range(len(x))) + b[i]
        for i in range(len(b))
    ]

def predict(angle_deg):
    rad = math.radians(angle_deg)
    x= [math.sin(rad), math.cos(rad)]
    x= relu(linear(x, W1, B1))
    x= relu(linear(x, W2, B2))
    x= linear(x, W3, B3)
    return x.index(max(x))

def show_arrow(direction_idx):
    pixels = ARROWS[direction_idx]
    hub.light_matrix.show_image(pixels)

# Main Loop
print("Classifier running")

last_direction = None

while True:
    angle = motor.absolute_position(port.A)    # absolute position: -180 to 179
    angle = (angle + 360) % 360# normalise to 0-359

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

    # Save a local copy
    # Get the absolute path of the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Safely join the directory path and the filename together
    file_path = os.path.join(script_dir, "hub_program.py")
    with open(file_path, "w", encoding="utf-8") as f:
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