"""
main.py — Trains the model, builds the hub program, uploads and runs it.

Run this file from VS Code after:
  pip install torch numpy bleak
"""

import asyncio
from TennisStandardClassifierV1 import train, extract_weights
import os
import torch

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

import time
from hub import motion_sensor, light_matrix, button
import gc

# ── Trained weights (baked in from PC training) ───────────────────────────────
W1 = {fmt(weights['w1'])}
B1 = {fmt(weights['b1'])}
W2 = {fmt(weights['w2'])}
B2 = {fmt(weights['b2'])}
W3 = {fmt(weights['w3'])}
B3 = {fmt(weights['b3'])}

SAMPLE_RATE_MS = 50
NUM_SAMPLES    = 30
GESTURES       = ['Forehand', 'Backhand', 'Overhead', 'None']

# neural net forward pass
def relu(x):
    return [v if v > 0 else 0.0 for v in x]

def linear(x, w, b):
    return [
        sum(w[i][j] * x[j] for j in range(len(x))) + b[i]
        for i in range(len(b))
    ]

def predict(x):
    x = relu(linear(x, W1, B1))
    x = relu(linear(x, W2, B2))
    x = linear(x, W3, B3)
    return x.index(max(x))

def collect_sample():
    sample = []
    for _ in range(NUM_SAMPLES):
        ax, ay, az = motion_sensor.acceleration(False)
        gx, gy, gz = motion_sensor.angular_velocity(False)
        sample.extend([ax, ay, az, gx, gy, gz])
        time.sleep_ms(SAMPLE_RATE_MS)
    return sample

# Main Loop
print("Ready. Press LEFT button to classify.")
light_matrix.write("?")

last_position = None

while True:
    # Wait for left button press
    while not button.pressed(button.LEFT):
        time.sleep_ms(50)
    # Wait for release
    while button.pressed(button.LEFT):
        time.sleep_ms(50)

    # Countdown
    for i in range(3, 0, -1):
        light_matrix.write(str(i))
        time.sleep(1)

    # Flash to signal go
    light_matrix.show([9] * 25)
    time.sleep_ms(200)
    light_matrix.show([0] * 25)

    # Collect and classify
    sample  = collect_sample()
    gesture = predict(sample)

    # Show first initial
    light_matrix.write(GESTURES[gesture][0])
    print(GESTURES[gesture])
    time.sleep(2)
    light_matrix.write("?")
"""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def round_weights(weights, decimals=12):
    if isinstance(weights, list):
        return [round_weights(w, decimals) for w in weights]
    return round(weights, decimals)

async def main():
    # Train model
    model   = train()
    model.verbose = True
    x = [659,-780,-47,-143,-81,8,680,-772,-6,-133,-126,39,666,-755,44,-108,-194,8,615,-745,99,-76,-292,30,539,-749,172,6,-476,103,440,-771,181,149,-776,179,361,-833,108,209,-1110,254,300,-913,-5,256,-1397,149,211,-921,-148,418,-1527,230,165,-968,-293,368,-1599,219,141,-994,-410,440,-1602,258,109,-992,-500,361,-1351,246,78,-1002,-521,84,-1263,122,83,-994,-566,18,-1048,77,92,-965,-610,30,-833,72,104,-943,-692,32,-560,61,110,-949,-837,105,-157,117,117,-996,-969,231,513,264,143,-1189,-1150,242,1344,156,95,-1357,-1224,269,2092,-131,-21,-1503,-1197,-163,2722,-830,-112,-1588,-1040,-736,2960,-1441,-105,-1583,-823,-1227,3049,-1926,-35,-1534,-606,-1600,2918,-2214,-4,-1426,-345,-1874,2744,-2364,-32,-1354,-80,-1866,2614,-2254,-169,-1197,373,-1422,2561,-2100,-171,-923,712,-995,1745,-2107,-16,-484,744,-483,853,-1680,169,-165,719,-221,455,-1114]
    x = torch.tensor(x, dtype=torch.float32)
    print("Before anything:")
    print(x)
    model(x)
    model.verbose = False

    weights = extract_weights(model)

    # Round down
    weights = {k: round_weights(v) for k, v in weights.items()}

    # Build SPIKE program
    program_source = build_hub_program(weights)

    # Save a local copy
    # Get the absolute path of the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Safely join the directory path and the filename together
    file_path = os.path.join(script_dir, "standard_tennis_hub.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(program_source)
    print("Hub program written to standard_tennis_hub.py")


if __name__ == "__main__":
    asyncio.run(main())