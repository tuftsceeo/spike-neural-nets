import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NeuralNetOnSPIKE.Hubs.spike import SpikeHub
def get_test_program():
    return f"""\
import hub
import time
from hub import light_matrix

hub.light_matrix.write("hi")
"""

def get_hub_program():
    return f"""\
# hub_data_collection.py — runs on the SPIKE Prime hub
# Records 1.5 seconds of IMU data and sends it over USB serial.
#
# Upload this manually via the SPIKE app or your existing upload script,
# then run it from there. The PC script listens for the output.
#
# Output format (one line per recording):
#SAMPLE,<label>,ax,ay,az,gx,gy,gz,ax,ay,az,gx,gy,gz,...
#(30 timesteps × 6 values = 180 comma-separated floats after the label)

import hub
import time
from hub import motion_sensor

SAMPLE_RATE_MS= 50 # one sample every 50ms
NUM_SAMPLES    = 30 # 30 × 50ms = 1.5 seconds
COUNTDOWN_SEC= 2    # seconds of countdown before recording

GESTURES = ["Forehand", "Backhand", "Overhead", "None"]

def countdown():
    #Show a countdown on the display so you know when to move.
    for i in range(COUNTDOWN_SEC, 0, -1):
        hub.light_matrix.write(str(i))
        time.sleep(1)
    # Flash the display to signal "go"
    hub.light_matrix.show([9] * 25)
    time.sleep(0.2)
    hub.light_matrix.show([0] * 25)

def record(label_idx):
    #Record NUM_SAMPLES IMU readings and print them as a CSV line.
    samples = []

    for _ in range(NUM_SAMPLES):
        ax, ay, az = motion_sensor.acceleration(False) # m/s²
        gx, gy, gz = motion_sensor.angular_velocity(False) # deg/s
        samples.extend([ax, ay, az, gx, gy, gz])
        time.sleep_ms(SAMPLE_RATE_MS)

    # Format: SAMPLE,<label_idx>,v0,v1,v2,...
    values = ",".join(str(round(v, 4)) for v in samples)
    print("SAMPLE," + "," + label_idx + "," + values)

    # Show a tick on the display to confirm recording saved
    hub.light_matrix.write("ok")
    time.sleep(1)
    hub.light_matrix.show([0] * 25)

def wait_for_button(idx):
    #Block until the centre hub button is pressed.
    hub.light_matrix.write(GESTURES[idx][0])
    while not hub.button.pressed(hub.button.LEFT):
        time.sleep_ms(50)
    # Wait for release so we don't double-trigger
    while hub.button.pressed(hub.button.LEFT):
        time.sleep_ms(50)

# ── Main loop ─────────────────────────────────────────────────────────────────

print("READY")# PC script waits for this before starting

gesture_idx = 0

while True:
    gesture_name = GESTURES[gesture_idx]

    # Prompt on display — cycle through gestures with button press
    hub.light_matrix.write(gesture_name[0])
    print("GESTURE," + str(gesture_idx) + "," + gesture_name)

    wait_for_button(gesture_idx)
    countdown()
    record(gesture_idx)

    # Cycle to the next gesture
    gesture_idx = (gesture_idx + 1) % len(GESTURES)
"""
async def main():
    async with SpikeHub() as hub:
        await hub.upload_program(get_test_program(), 2, "test_upload.py")
        await hub.stream_console()

if __name__ == "__main__":
    asyncio.run(main())