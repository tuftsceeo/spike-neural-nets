"""
Toy LSTM for the line-follower "gap" problem — hidden size 1, fully hand-traceable.

SENSOR MODEL (this is the part that matters physically):
  A single reflectance/color sensor hovers over the EDGE of the line (edge-following,
  not centered-on-line following). The raw sensor reports a reflection value from
  0 to 100 (0 = fully dark, 100 = fully light). Normalize it just by dividing by 100:

      raw = 0    ->  x_t = 0.0   -> fully over the dark line (drifted too far onto the line)
      raw = 50   ->  x_t = 0.5   -> straddling the edge correctly (this is the setpoint)
      raw = 100  ->  x_t = 1.0   -> fully over the light background (drifted too far off the line)

  (If your calibrated edge threshold isn't exactly at raw=50, adjust the setpoint
  accordingly -- but the /100 normalization itself is what keeps values in [0, 1]
  before they hit the gate nonlinearities, avoiding the tanh-saturation issue you
  already traced on the LEGO net.)

  THE KEY PHYSICAL FACT: a gap cut in the tape reads IDENTICALLY to "drifted far off
  onto the background" -- both are "100% light detected," i.e. raw=100, x_t=1.0. So
  the ambiguous input is the SATURATED extreme, not the setpoint. A gap is not "no
  signal," it's "the same signal as a large, real drift." A feedforward net has no
  way to tell these apart from x_t alone; an LSTM can, because its state still
  remembers whether it was gently approaching the edge or genuinely veering away
  just before the saturation hit.

This file has three parts:
  1. forward_step / forward_sequence -- the gate equations, one scalar at a time
  2. backward_sequence -- truncated BPTT, unrolled over the whole cached window
  3. a demo: train on one 5-step sequence containing a gap
"""

import json
import os
import legoeducation as le

import numpy as np

np.random.seed(0)


def load_run(path):
    """Load a recorded run and return (sensor_readings, motor_left, motor_right),
    each a list in timestep order."""
    with open(path, "r") as f:
        records = json.load(f)
    sensors = [r["sensor"] for r in records]
    motor_left = [r["motor"][0] for r in records]
    motor_right = [r["motor"][1] for r in records]
    return sensors, motor_left, motor_right


def make_dashed(sensor_values, rng, white_value=95, min_gap=3, max_gap=15, num_gaps=15):
    """
    This run was recorded over a SOLID line, so it has no gaps. Simulate a
    dashed line by stamping `num_gaps` randomly placed, randomly sized chunks
    of readings with `white_value` -- a fully-light reading, exactly what the
    sensor reports when it's over a gap in the tape (see module docstring).

    Returns a new list; `sensor_values` is left untouched so it can still be
    used as the "true" (un-gapped) target sequence.
    """
    corrupted = list(sensor_values)
    n = len(corrupted)
    for _ in range(num_gaps):
        gap_len = int(rng.integers(min_gap, max_gap + 1))
        start = int(rng.integers(0, max(1, n - gap_len)))
        for t in range(start, min(start + gap_len, n)):
            corrupted[t] = white_value
    print(str(corrupted))
    return corrupted


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class ScalarLSTMCell:
    """
    LSTM cell with input dim = 1, hidden dim = 1.
    Every weight is a scalar, so every intermediate value is one traceable number
    -- same spirit as your single-neuron backprop visualizer, just recurrent.
    """

    def __init__(self, lr=0.1):
        scale = 0.5
        # f is forget gate, x means it is the x (input) coefficient, w or b is weight or bias
        # i is input gate (stuff to add to longterm memory, potential longterm memory)
        # g is what to multiply input by (% potential longterm memory)
        # o is ouput (percent to add to small)
        self.w_fx, self.w_fh, self.b_f = np.random.randn(3) * scale 
        self.w_ix, self.w_ih, self.b_i = np.random.randn(3) * scale
        self.w_gx, self.w_gh, self.b_g = np.random.randn(3) * scale
        self.w_ox, self.w_oh, self.b_o = np.random.randn(3) * scale
        self.w_y, self.b_y = np.random.randn(2) * scale  # linear readout: y_t = w_y*h_t + b_y
        self.lr = lr

    def forward_step(self, x_t, h_prev, c_prev): # h_prev is the previous short term and c is long term
        f_t = sigmoid(self.w_fx * x_t + self.w_fh * h_prev + self.b_f)   # forget gate
        i_t = sigmoid(self.w_ix * x_t + self.w_ih * h_prev + self.b_i)   # input gate
        g_t = np.tanh(self.w_gx * x_t + self.w_gh * h_prev + self.b_g)  # candidate
        o_t = sigmoid(self.w_ox * x_t + self.w_oh * h_prev + self.b_o)  # output gate

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * np.tanh(c_t) # tanh is memory and o is percent of that to add to short term
        y_t = self.w_y * h_t + self.b_y

        cache = dict(x_t=x_t, h_prev=h_prev, c_prev=c_prev,
                     f_t=f_t, i_t=i_t, g_t=g_t, o_t=o_t,
                     c_t=c_t, h_t=h_t, y_t=y_t)
        return y_t, h_t, c_t, cache

    def forward_sequence(self, xs):
        h, c = 0.0, 0.0
        caches, ys = [], []
        for x_t in xs:
            y_t, h, c, cache = self.forward_step(x_t, h, c)
            ys.append(y_t)
            caches.append(cache)
        return ys, caches

    def backward_sequence(self, caches, targets):
        """
        Truncated BPTT over the whole cached sequence (here, truncation length =
        sequence length, since we keep it short enough to unroll fully by hand).
        """
        T = len(caches)
        grads = dict(w_fx=0., w_fh=0., b_f=0.,
                     w_ix=0., w_ih=0., b_i=0.,
                     w_gx=0., w_gh=0., b_g=0.,
                     w_ox=0., w_oh=0., b_o=0.,
                     w_y=0., b_y=0.)

        dh_next = 0.0   # dL/dh_{t+1}, arriving from the future
        dc_next = 0.0   # dL/dc_{t+1}, arriving from the future
        total_loss = 0.0

        for t in reversed(range(T)):
            cache = caches[t]
            target = targets[t]

            # ---- readout layer ----
            loss = 0.5 * (cache['y_t'] - target) ** 2
            total_loss += loss
            dy = (cache['y_t'] - target)                 # dL/dy_t
            grads['w_y'] += dy * cache['h_t']
            grads['b_y'] += dy

            # dL/dh_t has TWO sources: this step's own output loss, and h_{t+1}'s
            # dependence on h_t. This sum is the one genuinely new idea vs. a
            # feedforward backward pass.
            dh = dy * self.w_y + dh_next

            # ---- output gate & cell state ----
            tanh_c = np.tanh(cache['c_t'])
            do = dh * tanh_c
            dc = dh * cache['o_t'] * (1 - tanh_c ** 2) + dc_next

            # pre-activation gradients: sigmoid'(z) = s(1-s), tanh'(z) = 1 - t^2
            do_pre = do * cache['o_t'] * (1 - cache['o_t'])
            df_pre = (dc * cache['c_prev']) * cache['f_t'] * (1 - cache['f_t'])
            di_pre = (dc * cache['g_t']) * cache['i_t'] * (1 - cache['i_t'])
            dg_pre = (dc * cache['i_t']) * (1 - cache['g_t'] ** 2)

            grads['w_ox'] += do_pre * cache['x_t']; grads['w_oh'] += do_pre * cache['h_prev']; grads['b_o'] += do_pre
            grads['w_fx'] += df_pre * cache['x_t']; grads['w_fh'] += df_pre * cache['h_prev']; grads['b_f'] += df_pre
            grads['w_ix'] += di_pre * cache['x_t']; grads['w_ih'] += di_pre * cache['h_prev']; grads['b_i'] += di_pre
            grads['w_gx'] += dg_pre * cache['x_t']; grads['w_gh'] += dg_pre * cache['h_prev']; grads['b_g'] += dg_pre

            # gradients passed back to t-1
            dh_next = (df_pre * self.w_fh + di_pre * self.w_ih +
                       dg_pre * self.w_gh + do_pre * self.w_oh)
            dc_next = dc * cache['f_t']
            # ^ THE key line for teaching vanishing gradients: dc_next is dc_t
            #   scaled by f_t. If f_t is near 1 through the gap steps, gradient
            #   (and memory) survives the saturated-input stretch intact. If f_t
            #   is near 0, both the gradient and the memory of "gently
            #   approaching the edge, not veering off" get wiped out at that step.

        return total_loss, grads

    def update(self, grads, clip=5.0):
        # Long sequences (like a ~450-step real run) accumulate gradient across
        # every timestep, so a clip is needed to keep a single bad step from
        # blowing up the whole update -- the toy 5-step demo didn't need this.
        # Clip by the GLOBAL norm of the whole gradient vector, not per-parameter.
        total_norm = np.sqrt(sum(g ** 2 for g in grads.values()))
        if total_norm > clip:
            scale = clip / total_norm
            grads = {name: g * scale for name, g in grads.items()}
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - self.lr * g)


if __name__ == "__main__":
    m = le.DoubleMotor()
    c = le.ColorSensor()
    m.connect(card_serial="6235")
    c.connect(card_serial="6235")
    # Real recorded run, over a SOLID line -- no gaps in the raw data.
    data_path = os.path.join(os.path.dirname(__file__), "line_run_1")
    raw_sensor, motor_left, motor_right = load_run(data_path)

    # Punch random-width, randomly-placed gaps into a COPY of the run, stamped
    # with white_value=95 (fully-light / "off the line" reading) to simulate
    # what this same run would look like over a dashed line. `raw_sensor`
    # itself is left intact -- it's what the tape "truly" looked like, and
    # becomes the training target.
    rng = np.random.default_rng(42)
    dashed_sensor = make_dashed(raw_sensor, rng, white_value=95,
                                 min_gap=3, max_gap=15,
                                 num_gaps=max(1, len(raw_sensor) // 30))

    # x_t: what the robot actually sees (dashed / gappy). target: what the
    # correct edge-following reading would have been had the line been solid.
    # The LSTM has to learn to hold that memory through each gap instead of
    # reacting to the saturated x_t=0.95 reading as if it were a real drift.
    xs = [s / 100.0 for s in dashed_sensor]
    targets = [s / 100.0 for s in raw_sensor]

    gapped_steps = [t for t in range(len(raw_sensor)) if raw_sensor[t] != dashed_sensor[t]]
    print(f"loaded {len(raw_sensor)} timesteps from {data_path}")
    print(f"corrupted {len(gapped_steps)} of them ({100 * len(gapped_steps) / len(raw_sensor):.1f}%) "
          f"to white={95} to simulate a dashed line")

    # Much lower lr than the 5-step demo used -- gradients accumulate over the
    # whole ~450-step run, so 0.3 diverges here.
    cell = ScalarLSTMCell(lr=0.01)

    def total_loss(caches):
        return sum(0.5 * (cache['y_t'] - targets[t]) ** 2 for t, cache in enumerate(caches))

    print("\n=== forward pass BEFORE training (random weights) ===")
    _, caches = cell.forward_sequence(xs)
    print(f"total loss before training: {total_loss(caches):.5f}")

    print("\n=== training with full BPTT over the whole recorded run ===")
    for epoch in range(2000):
        _, caches = cell.forward_sequence(xs)
        loss, grads = cell.backward_sequence(caches, targets)
        cell.update(grads)
        if epoch % 250 == 0:
            print(f"epoch {epoch:4d}   total loss = {loss:.5f}")

    print("\n=== forward pass AFTER training (showing steps in/near a gap) ===")
    _, caches = cell.forward_sequence(xs)

    intervals = []
    for t in gapped_steps:
        if intervals and t == intervals[-1][1] + 1:
            intervals[-1][1] = t
        else:
            intervals.append([t, t])

    for start, end in intervals:
        lo, hi = max(0, start - 2), min(len(raw_sensor) - 1, end + 2)
        print(f"\n--- gap at t={start}..{end} ---")
        for t in range(lo, hi + 1):
            cache = caches[t]
            marker = " <-- gap" if raw_sensor[t] != dashed_sensor[t] else ""
            print(f"t={t:3d}  raw={raw_sensor[t]:3d}  dashed={dashed_sensor[t]:3d}  "
                  f"x={cache['x_t']:.2f}  f={cache['f_t']:.3f}  y_t={cache['y_t']:+.3f}  "
                  f"target={targets[t]:.3f}{marker}")

    print(f"\ntotal loss after training: {total_loss(caches):.5f}")
    print("\nCheck the forget gate f_t during each gap: values pulled toward 1 mean")
    print("the cell state -- and the memory of the true (solid-line) reading --")
    print("survives the saturated x_t=0.95 reading instead of being overwritten by it.")

    # ---- drive the real robot using the trained LSTM ----
    # In the recorded run, motor commands were an exact affine function of the
    # (solid-line) sensor reading: speedL = -0.15*sensor, speedR = 15 - 0.15*sensor.
    # Live, the raw reading can be gap-saturated (reads ~95 over a dash), so instead
    # of mapping the RAW reading straight to motor speeds, we feed it through the
    # trained cell one step at a time and map its held estimate of the TRUE sensor
    # position (y_t) to motor speeds instead -- this is the whole point of training
    # on simulated dashes: the cell state should keep coasting through a dash
    # instead of reacting to the saturated reading as a real drift.
    print("\n=== running live on the robot using the trained LSTM ===")
    h, cell_state = 0.0, 0.0  # persists across iterations so memory carries through dashes
    try:
        while True:
            read = c.sensor.reflection
            x_t = read / 100.0  # normalize it
            y_t, h, cell_state, _ = cell.forward_step(x_t, h, cell_state)
            predicted_sensor = float(np.clip(y_t, 0.0, 1.0)) * 100.0

            speedL = -0.15 * predicted_sensor
            speedR = 15 - 0.15 * predicted_sensor
            m.motor_run(speed=speedL, motor=le.MOTOR_LEFT)
            m.motor_run(speed=speedR, motor=le.MOTOR_RIGHT)
    except KeyboardInterrupt:
        m.motor_stop()
        m.disconnect()
        c.disconnect()

    