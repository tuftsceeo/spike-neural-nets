"""
Self-calibrating line follower with online backprop -- LIVE LEARNING DEMO.

The network starts on truly random weights, and the ONLY training it
ever gets is the handful of backprop steps it takes per tick, live,
while it's already driving. Expect it to spin, stall, or wander for
the first several seconds before it visibly starts correcting toward
the line as the weights converge.

Self-recalibration: cal.hi can only ever be pulled DOWN by a slow fixed
decay (never by a fresh low reading), so once the robot is driving
smoothly it may stop physically visiting the true dark/light extremes
on its own -- and if a period reset assumes some fixed "neutral" value
that doesn't match the current lighting, the same lag reappears from a
different starting point. So every WOBBLE_PERIOD_TICKS, this version
resets cal.lo/cal.hi to the CURRENT real sensor reading (not a fixed
guess), then spins the robot across the line edge to let real min/max
readings pull lo/hi apart from that grounded starting point in both
directions -- refreshing calibration against whatever lighting is
actually present right now, whether or not it changed.

Hardware note: tank drive (movement_move_tank) handles left/right
convention internally on this chassis, so no manual sign flip is
needed here (unlike the run_left/run_right version).
"""

import math
import random
import time
from collections import deque
import matplotlib.pyplot as plt
from lelib import doubleMotor, colorSensor

random.seed(5)

# How often (in ticks) to redraw the plot. Redrawing every tick will
# noticeably slow the control loop, so batch it.
PLOT_EVERY = 5
HISTORY_LEN = 400   # ticks of history shown on screen at once

# Small per-tick delay so a human can actually watch the behavior
# change, rather than it converging faster than the eye can follow.
TICK_SLEEP = 0.03

# ---- Forced re-exploration ----
# Every WOBBLE_PERIOD_TICKS, override the motors for WOBBLE_DURATION_TICKS
# and spin the robot in place across the line edge, purely to refresh
# cal.lo / cal.hi against real readings. Training and calibration continue
# normally during this window -- only the motor command source changes.
WOBBLE_PERIOD_TICKS = 400
WOBBLE_DURATION_TICKS = 60
WOBBLE_TURN_STRENGTH = 70     # motor magnitude while forced-exploring

# Safety floor: if decay ever shrinks the calibrated range faster than
# real exploration refreshes it, don't let lo/hi cross or collapse --
# clamp the gap open around the current midpoint instead.
MIN_GAP = 8.0

# Prevents the classic tanh-saturation failure mode: with online training
# running every tick forever and no regularization, w/b can grow large
# enough that tanh(w*x+b) saturates near +-1 across the whole realistic
# input range. Once that happens, the hidden layer's activations become
# nearly identical for every reading, the (1-h^2) derivative term in
# backprop collapses toward 0, and the network structurally can't express
# different behavior for different readings anymore -- no amount of
# further training (or correct recalibration) fixes it, because the
# gradient that would un-saturate it is itself the thing that vanished.
# Clipping keeps w/b in a range where tanh still has real slope.
WEIGHT_CLIP = 6.0


# =========================================================
# Self-calibration: running min/max with slow decay
# =========================================================
class Calibrator:
    DECAY = 0.05          # units per tick that lo/hi relax inward

    def __init__(self):
        self.lo, self.hi = 50.0, 50.0

    def update(self, r):
        # update the lo and hi if a higher or lower value comes into play
        self.lo = min(self.lo, r)
        self.hi = max(self.hi, r)
        self.lo += self.DECAY
        self.hi -= self.DECAY

        # Safety: decay (or a run of readings all on one side) could in
        # principle shrink the gap to nothing or cross it. Don't let the
        # self-labels degrade into nonsense -- hold a minimum gap open
        # around the current midpoint if that ever happens.
        if self.hi - self.lo < MIN_GAP:
            mid = 0.5 * (self.lo + self.hi)
            self.lo = mid - MIN_GAP / 2
            self.hi = mid + MIN_GAP / 2

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
            # Keep tanh's input weights in a range where it still has
            # real slope -- see WEIGHT_CLIP comment above.
            self.w[j] = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, self.w[j]))
            self.b[j] = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, self.b[j]))
        return 0.5 * (d_out[0] ** 2 + d_out[1] ** 2)


# =========================================================
# Self-labeling: the robot writes its own training data
# =========================================================
def self_labeled_batch(cal):
    return [
        (cal.lo,    0.40, 0),      # darkest it has seen -> brake, turn right
        (cal.mid(), 0.90, 0.90),   # midpoint            -> sprint
        (cal.hi,    0, 0.40),      # lightest it has seen -> brake, turn left
    ]


def clamp(v, lo=-100, hi=100):
    return max(lo, min(v, hi))


def drive(left_cmd, right_cmd):
    """Single place where commands reach the motors. Tank drive handles
    left/right convention internally on this chassis."""
    SCALING = 0.15
    motor.movement_move_tank(SCALING * clamp(left_cmd), SCALING * clamp(right_cmd))


# =========================================================
# Live plot of weights & biases, calibration bounds, and loss
# =========================================================
class LivePlot:
    def __init__(self, net, history_len=HISTORY_LEN):
        self.history_len = history_len
        self.t = 0
        self.ticks = deque(maxlen=history_len)

        self.series = {}
        for j in range(net.n):
            self.series["w[{}]".format(j)] = deque(maxlen=history_len)
            self.series["b[{}]".format(j)] = deque(maxlen=history_len)
        for k in range(2):
            for j in range(net.n):
                self.series["v[{}][{}]".format(k, j)] = deque(maxlen=history_len)
            self.series["c[{}]".format(k)] = deque(maxlen=history_len)
        self.loss_hist = deque(maxlen=history_len)

        self.lo_hist = deque(maxlen=history_len)
        self.mid_hist = deque(maxlen=history_len)
        self.hi_hist = deque(maxlen=history_len)

        plt.ion()
        self.fig, (self.ax_params, self.ax_cal, self.ax_loss) = plt.subplots(
            3, 1, figsize=(9, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 1]})

        self.lines = {}
        for name in self.series:
            (line,) = self.ax_params.plot([], [], label=name, linewidth=1.2)
            self.lines[name] = line
        self.ax_params.set_ylabel("value")
        self.ax_params.set_title("Weights & biases (live)")
        self.ax_params.legend(loc="upper left", ncol=4, fontsize=7)
        self.ax_params.axhline(0, color="gray", linewidth=0.5)

        (self.lo_line,) = self.ax_cal.plot([], [], label="cal.lo", color="tab:blue", linewidth=1.4)
        (self.mid_line,) = self.ax_cal.plot([], [], label="cal.mid()", color="gray", linewidth=1.0, linestyle="--")
        (self.hi_line,) = self.ax_cal.plot([], [], label="cal.hi", color="tab:orange", linewidth=1.4)
        self.ax_cal.set_ylabel("sensor units")
        self.ax_cal.set_title("Calibration bounds (live)")
        self.ax_cal.legend(loc="upper left", fontsize=8)

        (self.loss_line,) = self.ax_loss.plot([], [], color="black", linewidth=1.2)
        self.ax_loss.set_ylabel("loss")
        self.ax_loss.set_xlabel("tick")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)

    def push(self, net, loss, cal):
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
        self.lo_hist.append(cal.lo)
        self.mid_hist.append(cal.mid())
        self.hi_hist.append(cal.hi)

    def maybe_draw(self, force=False):
        if not force and self.t % PLOT_EVERY != 0:
            return
        xs = list(self.ticks)
        for name, line in self.lines.items():
            line.set_data(xs, list(self.series[name]))
        self.ax_params.relim()
        self.ax_params.autoscale_view()

        self.lo_line.set_data(xs, list(self.lo_hist))
        self.mid_line.set_data(xs, list(self.mid_hist))
        self.hi_line.set_data(xs, list(self.hi_hist))
        self.ax_cal.relim()
        self.ax_cal.autoscale_view()

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
sensor.connect(card_serial="6081")
motor = doubleMotor()
motor.connect(card_serial="6081")

net = Net()
cal = Calibrator()

print("Sweeping... move the sensor across the line edge now.")
SWEEP_TICKS = 60
time.sleep(1)
for i in range(SWEEP_TICKS):
    r = sensor.sensor.reflection
    cal.update(r)
    time.sleep(0.1)
print("  calibrated:  lo={:.1f}  mid={:.1f}  hi={:.1f}".format(
    cal.lo, cal.mid(), cal.hi))

print("Starting on RANDOM weights -- expect it to misbehave at first.")
plot = LivePlot(net)

# =========================================================
# Main loop
# =========================================================
tick = 0
while True:
    tick += 1
    r = sensor.sensor.reflection
    cal.update(r)

    loss = 0.0
    for sample in self_labeled_batch(cal):
        loss += net.train_step(*sample)
    plot.push(net, loss, cal)
    plot.maybe_draw()

    phase_in_period = tick % WOBBLE_PERIOD_TICKS
    if phase_in_period == 0:
        # Reset to the CURRENT real reading, not a fixed guess. Whatever
        # the lighting is right now, this is a grounded starting point --
        # the wobble below then pulls lo/hi apart from here based on
        # what's actually there, instead of assuming "50" is meaningful.
        cal.lo = r
        cal.hi = r
    if phase_in_period < WOBBLE_DURATION_TICKS:
        half = WOBBLE_DURATION_TICKS // 2
        turn = WOBBLE_TURN_STRENGTH if phase_in_period < half else -WOBBLE_TURN_STRENGTH
        drive(turn, -turn)
    else:
        _, _, y = net.forward(r)
        drive(100 * y[0], 100 * y[1])

    if TICK_SLEEP:
        time.sleep(TICK_SLEEP)