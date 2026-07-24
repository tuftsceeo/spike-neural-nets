import legoeducation as le
import json
import os

m = le.DoubleMotor()
c = le.ColorSensor()

m.connect(card_serial="6081")
c.connect(card_serial="6081")

FILE_NAME = "line_run_1"
samples = []

SCALE = 0.15
t = 0

try:
    while True:
        read = c.sensor.reflection
        speedL = -read * SCALE
        speedR = (100-read) * SCALE
        m.motor_run(speed=speedL, motor=le.MOTOR_LEFT)
        m.motor_run(speed=speedR, motor=le.MOTOR_RIGHT)

        samples.append({
            "t": t,
            "sensor": read,
            "motor": [speedL, speedR],
        })

        t += 1

except KeyboardInterrupt:
    c.disconnect()
    m.disconnect()

    with open(FILE_NAME, "w", encoding="utf-8") as file:
        file.write(json.dumps(samples) + "\n")
    print("saved")

