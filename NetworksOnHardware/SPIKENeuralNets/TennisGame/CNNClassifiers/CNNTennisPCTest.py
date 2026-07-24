# Just training and running on my PC to see if it actually works before I put on spike
# Below just change this import to test the various versions
from NeuralNetOnSPIKE.TennisGame.CNNClassifiers.TennisCNNClassifierV2 import train
import asyncio
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Hubs import Hub

sample_buffer = []
GESTURES     = ["Forehand", "Backhand", "Overhead", "None"]
NUM_TIMESTEPS = 30   # 30 × 50ms = 1.5 seconds
IMU_KEYS     = ["Ax", "Ay", "Az", "gyro_x", "gyro_y", "gyro_z"]

async def grab(message):
    imu = message.get("IMU")
    if not imu:
        return

    # Extract the 6 values we need in the right order
    row = [imu.get(k, 0) for k in IMU_KEYS]
    sample_buffer.append(row)

def predict(sample, model):
    # turn into tensor
    sample_tensor = torch.tensor(sample, dtype=torch.float32)

    # add a third dimension to sample
    sample_tensor = sample_tensor.unsqueeze(0)

    # Pass through model
    output = model(sample_tensor)

    return torch.argmax(output).item()

async def grab_data(hub):
    global sample_buffer
    sample_buffer.clear()
    print("Ready")
    await asyncio.sleep(1)
    print("Go")
    hub.collect = True
    await asyncio.sleep(1.5)   # collection window
    hub.collect = False

    # Check we got an okay sample
    if len(sample_buffer) != NUM_TIMESTEPS:
        print(f"  Warning: only got {len(sample_buffer)} samples, expected {NUM_TIMESTEPS}. Discarding.")
        return -1
    else:
        return 1

async def prompt_loop(hub, model):
    global sample_buffer
    repeat = True
    while repeat:
        try:
            print("Press Enter to record, 'q' to quit")
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, "> "
            )
            if user_input.strip().lower() == 'q':
                repeat = False
                await hub.ask()
            else:
                result = await grab_data(hub)
                if result < 0:
                    continue
                else:
                    output = predict(sample_buffer, model)
    
                    print("Classified as: " + GESTURES[output])

                    sample_buffer.clear()
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("ending")
            repeat = False
            await hub.ask()

async def main():
    # build and train a model
    print("training")
    model = train()
    print("done training")

    # Connect to spike
    spike = Hub.Hub_PS(hub = 0)
    await spike.ask()
    await spike.feed_rate(50)
    spike.final_callback = grab
    await prompt_loop(spike, model)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exited")

    
    
    
    


