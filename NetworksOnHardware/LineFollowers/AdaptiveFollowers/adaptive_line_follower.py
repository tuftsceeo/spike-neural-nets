"""
Self-calibrating line follower with online backprop.

The robot is never told what light values mean. Instead it:
  1. Discovers lo/hi by sweeping and tracking running min/max
     (the wobble keeps these fresh forever -- wobbling IS exploration)
  2. Generates its OWN training data from the calibration:
        lo  -> brake, turn right   (fell into the dark side)
        mid -> sprint              (on the edge)
        hi  -> brake, turn left    (fell onto the light side)
  3. Runs a few backprop steps every tick, forever.

When lighting changes, calibration drifts, the self-labels shift,
and gradient descent retrains the network live.

Pure Python, no numpy -- portable to the LEGO hub's MicroPython.
"""

import math
import random
from lelib import doubleMotor, colorSensor

random.seed(7)

# =========================================================
# Simulated world (replace with real sensor on the robot)
# =========================================================
class World:
    """Physical position p in [-1, +1] maps to a light reading.
    Lighting conditions can change via gain/offset."""
    def __init__(self):
        self.gain, self.offset = 1.0, 0.0   # normal lighting

    def read(self, p):
        base = 40 + 20 * p                  # 20 / 40 / 60 ground truth
        r = self.gain * base + self.offset + random.uniform(-1, 1)
        return max(0.0, min(100.0, r))

    def dim_the_lights(self):
        self.gain, self.offset = 0.5, 5.0   # readings become 15 / 25 / 35


# =========================================================
# Self-calibration: running min/max with slow decay
# =========================================================
class Calibrator:
    DECAY = 0.05          # units per tick that lo/hi relax inward

    def __init__(self):
        self.lo, self.hi = 50.0, 50.0

    def update(self, r):
        # real readings push the bounds out instantly...
        self.lo = min(self.lo, r)
        self.hi = max(self.hi, r)
        # ...and the bounds relax inward slowly, so they can re-adapt
        self.lo += self.DECAY
        self.hi -= self.DECAY

    def mid(self):
        return 0.5 * (self.lo + self.hi)


# =========================================================
# The network: raw reading -> 3 tanh hidden -> 2 motors
# =========================================================
class Net:
    def __init__(self, n_hidden=3, lr=0.3):
        self.n, self.lr = n_hidden, lr
        u = lambda: random.uniform(-1, 1)
        self.w = [u() for _ in range(n_hidden)]
        self.b = [u() for _ in range(n_hidden)]
        self.v = [[u() for _ in range(n_hidden)] for _ in range(2)]
        self.c = [u() for _ in range(2)]

    def forward(self, reading):
        x = reading / 100.0                       # fixed scaling only
        h = [math.tanh(self.w[j] * x + self.b[j]) for j in range(self.n)]
        y = [sum(self.v[k][j] * h[j] for j in range(self.n)) + self.c[k]
             for k in range(2)]
        return x, h, y

    def train_step(self, reading, t_L, t_R):
        x, h, y = self.forward(reading)
        d_out = [t_L - y[0], t_R - y[1]]
        d_hid = [(d_out[0] * self.v[0][j] + d_out[1] * self.v[1][j])
                 * (1.0 - h[j] ** 2) for j in range(self.n)]
        for k in range(2):
            for j in range(self.n):
                self.v[k][j] += self.lr * d_out[k] * h[j]
            self.c[k] += self.lr * d_out[k]
        for j in range(self.n):
            self.w[j] += self.lr * d_hid[j] * x
            self.b[j] += self.lr * d_hid[j]
        return 0.5 * (d_out[0] ** 2 + d_out[1] ** 2)


# =========================================================
# Self-labeling: the robot writes its own training data
# =========================================================
def self_labeled_batch(cal):
    return [
        (cal.lo,    0.80, 0.20),   # darkest it has seen -> brake, turn right
        (cal.mid(), 0.90, 0.90),   # midpoint            -> sprint
        (cal.hi,    0.20, 0.80),   # lightest it has seen-> brake, turn left
    ]


def behavior_table(world, net, label):
    print("  {}".format(label))
    for name, p in (("left of line ", -1.0), ("ON the line  ", 0.0),
                    ("right of line", +1.0)):
        r = world.read(p)
        _, _, y = net.forward(r)
        print("    {}  reading={:5.1f}  ->  L {:5.1f}   R {:5.1f}".format(
            name, r, 100 * y[0], 100 * y[1]))


# =========================================================
# The demo
# =========================================================
if __name__ == "__main__":
    world, cal, net = World(), Calibrator(), Net()

    # ---- Startup: sweep across the line to seed calibration ----
    print("PHASE 1: startup sweep + initial self-training")
    for i in range(60):                       # sweep p from -1 to +1
        p = -1.0 + 2.0 * i / 59
        cal.update(world.read(p))
    print("  discovered on its own:  lo={:.1f}  mid={:.1f}  hi={:.1f}"
          .format(cal.lo, cal.mid(), cal.hi))

    for _ in range(2000):                     # initial training burst
        for r, tL, tR in self_labeled_batch(cal):
            net.train_step(r, tL, tR)
    behavior_table(world, net, "behavior after self-training:")

    # ---- Lighting change! ----
    world.dim_the_lights()
    print("\nPHASE 2: *** LIGHTS DIMMED *** (true readings now ~15/25/35)")
    behavior_table(world, net, "behavior immediately after (confused):")

    # ---- Live adaptation: wobble, recalibrate, retrain, every tick ----
    print("\nPHASE 3: adapting online (calibrate + backprop every tick)")
    wobble = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5]   # crossing the edge
    for tick in range(1200):
        p = wobble[tick % len(wobble)]
        cal.update(world.read(p))
        loss = sum(net.train_step(r, tL, tR)
                   for r, tL, tR in self_labeled_batch(cal))
        if tick % 300 == 0 or tick == 1199:
            print("  tick {:4d}   lo={:5.1f} mid={:5.1f} hi={:5.1f}   "
                  "loss={:.4f}".format(tick, cal.lo, cal.mid(), cal.hi, loss))

    behavior_table(world, net, "behavior after adapting:")
    print("\n  Same weights-update rule, no human intervention: the robot")
    print("  re-discovered the light values and retrained itself.")


# ---- On the real robot (sketch) -------------------------------------------
# Startup: spin/arc slowly for ~2 s while calling cal.update(sensor.read())
# Main loop, every tick:
#     r = sensor.read()
#     cal.update(r)
#     for sample in self_labeled_batch(cal):      # 3 tiny backprop steps
#         net.train_step(*sample)
#     _, _, y = net.forward(r)
#     motor_L.run(clamp(100 * y[0])); motor_R.run(clamp(100 * y[1]))
# Tune Calibrator.DECAY to taste: higher = adapts faster, but less stable.
sensor = colorSensor()
sensor.connect(card_serial="1003")
motor = doubleMotor()
motor.connect(card_serial="1003")
while True:
    r = sensor.sensor.reflection
    cal.update(r)
    for sample in self_labeled_batch(cal):      # 3 tiny backprop steps
        net.train_step(*sample)
    _, _, y = net.forward(r)
    motor.run_left(min((100 * y[0]), 100))
    motor.run_right(min((100 * y[1]), 100))


