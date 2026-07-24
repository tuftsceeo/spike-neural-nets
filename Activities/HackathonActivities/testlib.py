# import lelib
# from lelib import colorSensor, doubleMotor
# cs = colorSensor()
# cs.connect(card_serial="1131")

# dm = doubleMotor()
# dm.connect(card_serial="1131")

# curr_speed = 0
# while True:
#     curr_speed = 0.1 * (100 - cs.sensor.reflection)
#     dm.set_speed(curr_speed)
#     dm.run()

# import lelib
# from lelib import controller, doubleMotor
# import time
# c = controller()
# dm = doubleMotor()

# c.connect(card_serial="1131")
# dm.connect(card_serial="1131")

# def is_active_left(controller):
#     return controller.sensor.leftPercent > 5 or controller.sensor.leftPercent < -5

# def is_active_right(controller):
#     return controller.sensor.rightPercent > 5 or controller.sensor.rightPercent < -5

# def binary_step(x):
#     if x <= 0:
#         return 0
#     else:
#         return 1

# def layer(x1, x2):
#     return (x1 + x2)

# def predict(x1, x2):
#     return (binary_step(layer(x1, x2)))

# while True:
#     go = predict(int(is_active_left(c)), int(is_active_right(c)))
#     if go == 1:
#         print("going")
#         dm.run()
#     else:
#         print("stopping")
#         dm.stop()
#     time.sleep(0.5)

import lelib
from lelib import controller, doubleMotor
import time
c = controller()
dm = doubleMotor()

c.connect(card_serial="1131")
dm.connect(card_serial="1131")

def is_active_left(controller):
    return controller.sensor.leftPercent > 5 or controller.sensor.leftPercent < -5

def is_active_right(controller):
    return controller.sensor.rightPercent > 5 or controller.sensor.rightPercent < -5

def binary_step(x):
    if x <= 0:
        return 0
    else:
        return 1
    
def ReLU(x):
    if x < 0:
        return 0
    else:
        return x

def layer1_1(x1, x2):
    return -x1 - x2 + 2

def layer1_2(x1, x2):
    return x1 + x2

def layer2(x1, x2):
    return x1 + x2 - 1

def predict(x1, x2):
    out1 = binary_step(layer1_1(x1, x2))
    out2 = binary_step(layer1_2(x1, x2))
    out = ReLU(layer2(out1, out2))
    return out

while True:
    go = predict(int(is_active_left(c)), int(is_active_right(c)))
    if go == 1:
        print("going")
        dm.run()
    else:
        print("stopping")
        dm.stop()