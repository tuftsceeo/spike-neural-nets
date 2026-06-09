import asyncio
import json
import csv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Hubs import Hub

# ── Configuration ─────────────────────────────────────────────────────────────

GESTURES     = ["Forehand", "Backhand", "Overhead", "None"]
NUM_TIMESTEPS = 30   # 30 × 50ms = 1.5 seconds
IMU_KEYS     = ["Ax", "Ay", "Az", "gyro_x", "gyro_y", "gyro_z"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "gesture_data")
sample_buffer = []   # accumulates IMU readings during collection window
current_gesture_idx = 0

# ── CSV header ────────────────────────────────────────────────────────────────

def make_header():
    cols = []
    for t in range(NUM_TIMESTEPS):
        for key in IMU_KEYS:
            cols.append(f"t{t}_{key}")
    return cols

def ensure_files():
    """Create output folder and CSV files with headers if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for gesture in GESTURES:
        path = os.path.join(OUTPUT_DIR, f"{gesture}.csv")
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(make_header())

# ── Per-sample buffer ─────────────────────────────────────────────────────────

async def grab(message):
    imu = message.get("IMU")
    if not imu:
        return

    # Extract the 6 values we need in the right order
    row = [imu.get(k, 0) for k in IMU_KEYS]
    sample_buffer.append(row)

def save_sample():
    """Flatten the buffer and append to the current gesture's CSV."""
    path = os.path.join(OUTPUT_DIR, f"{GESTURES[current_gesture_idx]}.csv")
    flat = [val for timestep in sample_buffer for val in timestep]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(flat)
    print(f"  Saved sample to {GESTURES[current_gesture_idx]}.csv  "
          f"(row length: {len(flat)})")

async def grab_data(hub):
    global sample_buffer
    sample_buffer.clear()
    print(f"\nGesture: {GESTURES[current_gesture_idx]}")
    print("Ready")
    await asyncio.sleep(1)
    print("Go")
    hub.collect = True
    await asyncio.sleep(1.5)   # collection window
    hub.collect = False

    # Save whatever we collected
    if len(sample_buffer) < NUM_TIMESTEPS:
        print(f"  Warning: only got {len(sample_buffer)} samples, expected {NUM_TIMESTEPS}. Discarding.")
        return

    save_sample()
    print(f"  Saved {len(sample_buffer)} samples to {GESTURES[current_gesture_idx]}.csv")
    sample_buffer.clear()

async def prompt_loop(hub):
    global current_gesture_idx
    ensure_files()
    repeat = True
    while repeat:
        try:
            counts = {g: count_rows(g) for g in GESTURES}
            print("\nCurrent counts:")
            for g, c in counts.items():
                print(f"  {g}: {c}")
            print(f"\nNext gesture: {GESTURES[current_gesture_idx]}")
            print("Press Enter to record, 'n' to switch gesture, 'q' to quit")
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, "> "
            )
            if user_input.strip().lower() == 'q':
                repeat = False
                await hub.ask()
            elif user_input.strip().lower() == 'n':
                current_gesture_idx = (current_gesture_idx + 1) % len(GESTURES)
                print(f"Switched to: {GESTURES[current_gesture_idx]}")
            else:
                await grab_data(hub)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("ending")
            repeat = False
            await hub.ask()

def count_rows(gesture):
    """Count how many samples already saved for this gesture."""
    path = os.path.join(OUTPUT_DIR, f"{gesture}.csv")
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header row

async def main():
    spike = Hub.Hub_PS(hub = 0)  # hub0 = spike
    print("asking")
    await spike.ask()
    await spike.feed_rate(50)
    spike.final_callback = grab
    await prompt_loop(spike)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited")