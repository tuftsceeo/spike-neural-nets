"""
build_hub_program.py
Trains the gesture classifier, extracts weights, and writes a self-contained
MicroPython program to hub_program.py. Paste that file into the SPIKE app.
"""

import os
import sys
import torch
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TennisCNNClassifierV3 import train, extract_weights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "v3_hub_cnn_program.py")

GESTURES = ["Forehand", "Backhand", "Overhead", "None"]


# ══════════════════════════════════════════════════════════════════════════════
# Hub program builder
# ══════════════════════════════════════════════════════════════════════════════

def round_weights(weights, decimals=6):
    """Recursively round all floats in a nested list structure."""
    if isinstance(weights, list):
        return [round_weights(w, decimals) for w in weights]
    return round(weights, decimals)

def fmt(lst):
    """Format a nested list compactly for embedding in source code."""
    if isinstance(lst[0], list):
        rows = ",\n        ".join(str(row) for row in lst)
        return f"[\n        {rows}\n    ]"
    return str(lst)


def build_hub_program(weights: dict) -> str:
    return f"""\
# hub_program.py — auto-generated, runs on SPIKE Prime hub
# Press LEFT button to classify a 1.5 second gesture window.
# Displays first initial of classified gesture on the light matrix.

import gc
import time
from hub import motion_sensor, light_matrix, button

# ── Trained weights ───────────────────────────────────────────────────────────

# conv1
WC1 = {fmt(weights['wc1'])}
BC1 = {fmt(weights['bc1'])}

# conv2
WC2 = {fmt(weights['wc2'])}
BC2 = {fmt(weights['bc2'])}

# fc
WF = {fmt(weights['wf'])}
BF = {fmt(weights['bf'])}

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE_MS = 50
NUM_SAMPLES    = 30
GESTURES       = {GESTURES}

# ── Pure-Python forward pass ──────────────────────────────────────────────────
def conv1d(x, weight, bias, kernel_size):
    \"\"\"
    x:      list of shape (time, in_channels)
    weight: list of shape (out_channels, in_channels, kernel_size)
    bias:   list of shape (out_channels,)
    returns list of shape (time - kernel_size + 1, out_channels)
    \"\"\"
    out_len      = len(x) - kernel_size + 1
    out_channels = len(weight)
    in_channels  = len(x[0])
    out = []
    for t in range(out_len):
        frame = []
        gc.collect()
        for oc in range(out_channels):
            val = bias[oc]
            for k in range(kernel_size):
                for ic in range(in_channels):
                    val += x[t + k][ic] * weight[oc][ic][k]
            frame.append(val)
        out.append(frame)
    return out

def global_avg_pool(x):
    \"\"\"
    x:       list of shape (time, channels)
    returns: list of shape (channels,) — mean across time
    \"\"\"
    num_channels = len(x[0])
    num_time     = len(x)
    return [
        sum(x[t][c] for t in range(num_time)) / num_time
        for c in range(num_channels)
    ]

def linear(x, weight, bias):
    \"\"\"
    x:      list of shape (in_features,)
    weight: list of shape (out_features, in_features)
    bias:   list of shape (out_features,)
    returns list of shape (out_features,)
    \"\"\"
    return [
        sum(weight[i][j] * x[j] for j in range(len(x))) + bias[i]
        for i in range(len(bias))
    ]

def relu(x):
    i = 0
    for row in x:
        j = 0
        for v in row:
            x[i][j] = max(0.0, v)
            j += 1
        gc.collect()
        i += 1
    return x

def predict(sample):
    \"\"\"
    sample: list of shape (30, 6) — 30 timesteps, 6 IMU values each
    returns: int — index of predicted gesture class
    \"\"\"
    x = sample                          # (30, 6)
    x = conv1d(x, WC1, BC1, 5)         # (26, 32)
    x = [[max(0.0, v) for v in row] for row in x]  # relu
    x = conv1d(x, WC2, BC2, 5)         # (22, 64)
    x = [[max(0.0, v) for v in row] for row in x]  # relu
    x = global_avg_pool(x)             # (64,)
    x = linear(x, WF, BF)             # (4,)
    return x.index(max(x))

# ── IMU collection ────────────────────────────────────────────────────────────

def collect_sample():
    \"\"\"Collect NUM_SAMPLES IMU readings and return as (30, 6) list.\"\"\"
    sample = []
    for _ in range(NUM_SAMPLES):
        ax, ay, az = motion_sensor.acceleration(False)
        gx, gy, gz = motion_sensor.angular_velocity(False)
        sample.append([ax, ay, az, gx, gy, gz])
        time.sleep_ms(SAMPLE_RATE_MS)
    return sample

# ── Main loop ─────────────────────────────────────────────────────────────────

print("Ready. Press LEFT button to classify.")
light_matrix.write("?")

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
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Training model...")
    model = train()
    model.eval()
    print("Training done.")

    weights = extract_weights(model)
    # Round way down
    weights = {k: round_weights(v) for k, v in weights.items()}

    print(f"Weight sizes:")
    for k, v in weights.items():
        size_kb = len(json.dumps(v)) / 1024
        print(f"  {k}: {size_kb:.1f} KB")

    program = build_hub_program(weights)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(program)

    total_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nHub program written to: {OUTPUT_PATH}")
    print(f"Total file size: {total_kb:.1f} KB")
    print("Paste the contents of hub_program.py into the SPIKE app.")


if __name__ == "__main__":
    main()