from lelib import singleMotor
import legoeducation as le
import time

m = singleMotor()
m.connect("5186")

starting_pos = m.motor.position

while starting_pos == m.motor.position:
    time.sleep(0.1)

last_pos = starting_pos

while last_pos != m.motor.position:
    last_pos = m.motor.position
    time.sleep(0.1)

# Now countdown:
degrees = last_pos - starting_pos
seconds = int(degrees/6)

for i in range(seconds):
    m.motor_run_for_degrees(6, blocking=False)
    time.sleep(1)

m.beep(pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=440)




