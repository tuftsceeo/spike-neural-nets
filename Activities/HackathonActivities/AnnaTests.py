import legoeducation as le
import time
# ACTIVITY 1 CODE SOLUTION

# c = le.ColorSensor()
# c.connect()

# m = le.DoubleMotor()
# m.connect()

# curr = 0
# while True:
#     curr = 0.1 * (100 - c.sensor.reflection)
#     m.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=curr, motor = le.MOTOR_LEFT)
#     m.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, speed=curr, motor = le.MOTOR_RIGHT)

# ACTIVITY 2 SOLUTION
# c = le.Controller()
# m = le.DoubleMotor()

# c.connect()
# m.connect()

# def abs(x):
#     if x < 0:
#         return -1 * x
#     else:
#         return x

# def binary_step(x):
#     if x <= 0:
#         return 0
#     else:
#         return 1

# def layer(x1, x2):
#     return (abs(x1) + abs(x2))

# def predict(x1, x2):
#     x1 = abs(x1)
#     x2 = abs(x2)
#     return (binary_step(layer(x1, x2)))

# while True:
#     go = predict(c.sensor.leftPercent, c.sensor.rightPercent)
#     if go == 1:
#         print("going")
#         m.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=10, motor = le.MOTOR_LEFT)
#         m.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, speed=10, motor = le.MOTOR_RIGHT)
#     else:
#         print("stopping")
#         m.motor_stop(motor=le.MOTOR_BOTH)
#     time.sleep(0.5)

# ACTIVITY 3
c = le.Controller()
m = le.DoubleMotor()
c.connect()
m.connect()

def abs(x):
    if x < 0:
        return -1 * x
    else:
        return x

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
    x1 = binary_step(abs(x1))
    x2 = binary_step(abs(x2))
    out1 = binary_step(layer1_1(x1, x2))
    out2 = binary_step(layer1_2(x1, x2))
    out = ReLU(layer2(out1, out2))
    return out

while True:
    go = predict(c.sensor.leftPercent, c.sensor.rightPercent)
    if go == 1:
        m.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=10, motor = le.MOTOR_LEFT)
        m.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, speed=10, motor = le.MOTOR_RIGHT)
    else:
        m.motor_stop(motor=le.MOTOR_BOTH)