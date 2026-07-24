import json
import os
import legoeducation as le

import numpy as np

np.random.seed(0)


# DATA READ IN STUFF
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
        scale = 0.5 # need to scale at first to make sure they dont immediately saturate tanh or sigmoid
        # f is forget gate, x means it is the x (input) coefficient, w or b is weight or bias
        # i is input gate (stuff to add to longterm memory, potential longterm memory)
        # g is what to multiply input by (% potential longterm memory)
        # o is ouput (percent to add to small)
        self.w_fx, self.w_fh, self.b_f = np.random.randn(3) * scale 
        self.w_ix, self.w_ih, self.b_i = np.random.randn(3) * scale
        self.w_gx, self.w_gh, self.b_g = np.random.randn(3) * scale
        self.w_ox, self.w_oh, self.b_o = np.random.randn(3) * scale
        
        # output layer, one neuron for each speed
        self.w_yL, self.b_yL = np.random.randn(2) * scale  # y_L = w_yL*h_t + b_yL
        self.w_yR, self.b_yR = np.random.randn(2) * scale  # y_R = w_yR*h_t + b_yR
        self.lr = lr

    def forward_step(self, x_t, h_prev, c_prev): # h_prev is the previous short term and c is long term
        f_t = sigmoid(self.w_fx * x_t + self.w_fh * h_prev + self.b_f)   # forget gate
        i_t = sigmoid(self.w_ix * x_t + self.w_ih * h_prev + self.b_i)   # input gate
        g_t = np.tanh(self.w_gx * x_t + self.w_gh * h_prev + self.b_g)  # candidate
        o_t = sigmoid(self.w_ox * x_t + self.w_oh * h_prev + self.b_o)  # output gate

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * np.tanh(c_t) # tanh is memory and o is percent of that to add to short term
        y_L = self.w_yL * h_t + self.b_yL
        y_R = self.w_yR * h_t + self.b_yR

        cache = dict(x_t=x_t, h_prev=h_prev, c_prev=c_prev,
                     f_t=f_t, i_t=i_t, g_t=g_t, o_t=o_t,
                     c_t=c_t, h_t=h_t, y_L=y_L, y_R=y_R)
        return (y_L, y_R), h_t, c_t, cache

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
        `targets` is a list of (target_L, target_R) pairs, one per timestep.
        """
        # Previous saved data
        T = len(caches)
        # initialize gradients at 0
        grads = dict(w_fx=0., w_fh=0., b_f=0.,
                     w_ix=0., w_ih=0., b_i=0.,
                     w_gx=0., w_gh=0., b_g=0.,
                     w_ox=0., w_oh=0., b_o=0.,
                     w_yL=0., b_yL=0.,
                     w_yR=0., b_yR=0.)

        dh_next = 0.0   # dL/dh_{t+1}, arriving from the future
        dc_next = 0.0   # dL/dc_{t+1}, arriving from the future
        total_loss = 0.0

        for t in reversed(range(T)): # BPTT (going backwards through time to backpropagate)
            # save current time stamp vars
            cache = caches[t] 
            target_L, target_R = targets[t]

            # ---- readout layer (two independent heads) ----
            # calculate loss/derivative for all vars in readout layer
            loss = 0.5 * (cache['y_L'] - target_L) ** 2 + 0.5 * (cache['y_R'] - target_R) ** 2
            total_loss += loss
            dyL = cache['y_L'] - target_L            # dL/dy_L
            dyR = cache['y_R'] - target_R             # dL/dy_R
            grads['w_yL'] += dyL * cache['h_t']
            grads['b_yL'] += dyL
            grads['w_yR'] += dyR * cache['h_t']
            grads['b_yR'] += dyR

            # dL/dh_t has THREE sources now: each readout head's own loss, plus
            # h_{t+1}'s dependence on h_t.
            dh = dyL * self.w_yL + dyR * self.w_yR + dh_next

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

    def update(self, grads, clip=50.0):
        # Long sequences (like a ~450-step real run) accumulate gradient across
        # every timestep, so a clip is needed to keep a single bad step from
        # blowing up the whole update.
        # Clip by the GLOBAL norm of the whole gradient vector, not per-parameter:
        # the readout gradients (fitting raw motor speeds) run 10-1000x larger
        # than the gate gradients, so an independent per-parameter clip would
        # squash them all to the same step size and the gates would barely move.
        total_norm = np.sqrt(sum(g ** 2 for g in grads.values()))
        if total_norm > clip:
            scale = clip / total_norm
            grads = {name: g * scale for name, g in grads.items()}
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - self.lr * g)


if __name__ == "__main__":
    m = le.DoubleMotor()
    c = le.ColorSensor()
    m.connect(card_serial="6081")
    c.connect(card_serial="6081")

    # Get data for solid line
    data_path = os.path.join(os.path.dirname(__file__), "line_run_1")
    raw_sensor, motor_left, motor_right = load_run(data_path)

    # Dashify it
    rng = np.random.default_rng(42)
    dashed_sensor = make_dashed(raw_sensor, rng, white_value=95,
                                 min_gap=3, max_gap=15,
                                 num_gaps=max(1, len(raw_sensor) // 30))

    # xs is made of x_ts where each is a sensor reading (normalized to 0-1 range)
    xs = [s / 100.0 for s in dashed_sensor]

    # targets were the correct motor speeds for each x_t
    targets = list(zip(motor_left, motor_right))

    # array of the spots that were dashified
    gapped_steps = [t for t in range(len(raw_sensor)) if raw_sensor[t] != dashed_sensor[t]]

    # the model
    cell = ScalarLSTMCell(lr=0.01)

    print("\n=== training with full BPTT over the whole recorded run ===")
    for epoch in range(2000):
        _, caches = cell.forward_sequence(xs)
        loss, grads = cell.backward_sequence(caches, targets)
        cell.update(grads)
        if epoch % 250 == 0:
            print(f"epoch {epoch:4d}   total loss = {loss:.5f}")

    # ---- drive the real robot using the trained LSTM ----
    print("\n=== running live on the robot using the trained LSTM ===")
    h, cell_state = 0.0, 0.0  # persists across iterations so memory carries through dashes
    try:
        while True:
            read = c.sensor.reflection
            x_t = read / 100.0  # normalize it
            (y_L, y_R), h, cell_state, _ = cell.forward_step(x_t, h, cell_state)
            speedL = float(y_L)
            speedR = float(y_R)
            m.motor_run(speed=speedL, motor=le.MOTOR_LEFT)
            m.motor_run(speed=speedR, motor=le.MOTOR_RIGHT)
    except KeyboardInterrupt:
        m.motor_stop()
        m.disconnect()
        c.disconnect()

    