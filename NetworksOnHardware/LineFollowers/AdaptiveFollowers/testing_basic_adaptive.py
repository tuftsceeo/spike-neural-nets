"""
Self-calibrating line follower with online backprop -- HARDWARE-ONLY VERSION.

No simulation, no pretraining. The robot starts with a randomly
initialized network and an uncalibrated sensor, physically sweeps
across the line once to seed calibration, then runs forever:
every tick it re-calibrates, re-generates its own training labels
from the current calibration, takes a few backprop steps, and drives.
"""

import math
import random
from collections import deque
import matplotlib.pyplot as plt
from lelib import doubleMotor, colorSensor
import time

random.seed(7)

# How often (in ticks) to redraw the plot. Redrawing every tick will
# noticeably slow the control loop, so batch it.
PLOT_EVERY = 5
HISTORY_LEN = 400   # ticks of history shown on screen at once


# =========================================================
# Self-calibration: running min/max with slow decay
# =========================================================
class Calibrator:
    DECAY = 0.05          # units per tick that lo/hi relax inward

    def __init__(self):
        self.lo, self.hi = 50.0, 50.0

    def update(self, r):
        self.lo = min(self.lo, r)
        self.hi = max(self.hi, r)
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
        x = reading / 100.0
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


def clamp(v, lo=-100, hi=100):
    return max(lo, min(v, hi))


# =========================================================
# Live plot of weights & biases as they train
# =========================================================
class LivePlot:
    """Rolling line plot of every trainable parameter, updated in place.

    Call .push(net) once per tick (cheap: just appends to deques).
    Call .maybe_draw() periodically (does the actual matplotlib redraw).
    """

    def __init__(self, net, history_len=HISTORY_LEN):
        self.history_len = history_len
        self.t = 0
        self.ticks = deque(maxlen=history_len)

        # One deque per scalar parameter, named for the legend.
        self.series = {}
        for j in range(net.n):
            self.series["w[{}]".format(j)] = deque(maxlen=history_len)
            self.series["b[{}]".format(j)] = deque(maxlen=history_len)
        for k in range(2):
            for j in range(net.n):
                self.series["v[{}][{}]".format(k, j)] = deque(maxlen=history_len)
            self.series["c[{}]".format(k)] = deque(maxlen=history_len)
        self.loss_hist = deque(maxlen=history_len)

        plt.ion()
        self.fig, (self.ax_params, self.ax_loss) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})

        self.lines = {}
        for name in self.series:
            (line,) = self.ax_params.plot([], [], label=name, linewidth=1.2)
            self.lines[name] = line
        self.ax_params.set_ylabel("value")
        self.ax_params.set_title("Weights & biases (live)")
        self.ax_params.legend(loc="upper left", ncol=4, fontsize=7)
        self.ax_params.axhline(0, color="gray", linewidth=0.5)

        (self.loss_line,) = self.ax_loss.plot([], [], color="black", linewidth=1.2)
        self.ax_loss.set_ylabel("loss")
        self.ax_loss.set_xlabel("tick")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)

    def push(self, net, loss):
        self.t += 1
        self.ticks.append(self.t)
        for j in range(net.n):
            self.series["w[{}]".format(j)].append(net.w[j])
            self.series["b[{}]".format(j)].append(net.b[j])
        for k in range(2):
            for j in range(net.n):
                self.series["v[{}][{}]".format(k, j)].append(net.v[k][j])
            self.series["c[{}]".format(k)].append(net.c[k])
        self.loss_hist.append(loss)

    def maybe_draw(self, force=False):
        if not force and self.t % PLOT_EVERY != 0:
            return
        xs = list(self.ticks)
        for name, line in self.lines.items():
            line.set_data(xs, list(self.series[name]))
        self.ax_params.relim()
        self.ax_params.autoscale_view()

        self.loss_line.set_data(xs, list(self.loss_hist))
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# =========================================================
# Hardware setup
# =========================================================
sensor = colorSensor()
sensor.connect(card_serial="1003")
motor = doubleMotor()
motor.connect(card_serial="1003")

net = Net()
cal = Calibrator()

# ---- Startup sweep: seed calibration from REAL readings ----
# Physically rock/rotate the sensor across the line's edge while this runs,
# so cal.lo / cal.hi capture the true dark and light extremes.
print("Sweeping... move the sensor across the line edge now.")
time.sleep(1)
SWEEP_TICKS = 60
for i in range(SWEEP_TICKS):
    r = sensor.sensor.reflection
    cal.update(r)
    time.sleep(0.01)
print("  calibrated:  lo={:.1f}  mid={:.1f}  hi={:.1f}".format(
    cal.lo, cal.mid(), cal.hi))

# ---- Initial training burst so it's not driving on random weights ----
print("Initial training burst...")
plot = LivePlot(net)
for i in range(2000):
    loss = 0.0
    for r, t_L, t_R in self_labeled_batch(cal):
        loss += net.train_step(r, t_L, t_R)
    plot.push(net, loss)
    if i % 20 == 0:      # thin out redraws during the fast initial burst
        plot.maybe_draw()
plot.maybe_draw(force=True)
print("  done. Starting main loop.")

# =========================================================
# Main loop: calibrate + train + drive, every tick, forever
# =========================================================
while True:
    r = sensor.sensor.reflection
    cal.update(r)
    loss = 0.0
    for sample in self_labeled_batch(cal):
        loss += net.train_step(*sample)
    plot.push(net, loss)
    plot.maybe_draw()
    _, _, y = net.forward(r)
    motor.run_left(clamp(100 * y[0]))
    motor.run_right(-clamp(100 * y[1]))