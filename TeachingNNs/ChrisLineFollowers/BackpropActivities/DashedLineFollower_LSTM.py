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

import numpy as np

np.random.seed(0)


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
            c = caches[t]
            target = targets[t]

            # ---- readout layer ----
            loss = 0.5 * (c['y_t'] - target) ** 2
            total_loss += loss
            dy = (c['y_t'] - target)                 # dL/dy_t
            grads['w_y'] += dy * c['h_t']
            grads['b_y'] += dy

            # dL/dh_t has TWO sources: this step's own output loss, and h_{t+1}'s
            # dependence on h_t. This sum is the one genuinely new idea vs. a
            # feedforward backward pass.
            dh = dy * self.w_y + dh_next

            # ---- output gate & cell state ----
            tanh_c = np.tanh(c['c_t'])
            do = dh * tanh_c
            dc = dh * c['o_t'] * (1 - tanh_c ** 2) + dc_next

            # pre-activation gradients: sigmoid'(z) = s(1-s), tanh'(z) = 1 - t^2
            do_pre = do * c['o_t'] * (1 - c['o_t'])
            df_pre = (dc * c['c_prev']) * c['f_t'] * (1 - c['f_t'])
            di_pre = (dc * c['g_t']) * c['i_t'] * (1 - c['i_t'])
            dg_pre = (dc * c['i_t']) * (1 - c['g_t'] ** 2)

            grads['w_ox'] += do_pre * c['x_t']; grads['w_oh'] += do_pre * c['h_prev']; grads['b_o'] += do_pre
            grads['w_fx'] += df_pre * c['x_t']; grads['w_fh'] += df_pre * c['h_prev']; grads['b_f'] += df_pre
            grads['w_ix'] += di_pre * c['x_t']; grads['w_ih'] += di_pre * c['h_prev']; grads['b_i'] += di_pre
            grads['w_gx'] += dg_pre * c['x_t']; grads['w_gh'] += dg_pre * c['h_prev']; grads['b_g'] += dg_pre

            # gradients passed back to t-1
            dh_next = (df_pre * self.w_fh + di_pre * self.w_ih +
                       dg_pre * self.w_gh + do_pre * self.w_oh)
            dc_next = dc * c['f_t']
            # ^ THE key line for teaching vanishing gradients: dc_next is dc_t
            #   scaled by f_t. If f_t is near 1 through the gap steps, gradient
            #   (and memory) survives the saturated-input stretch intact. If f_t
            #   is near 0, both the gradient and the memory of "gently
            #   approaching the edge, not veering off" get wiped out at that step.

        return total_loss, grads

    def update(self, grads):
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - self.lr * g)


if __name__ == "__main__":
    # Raw reflectance readings, 0-100 scale, normalized by dividing by 100:
    #   0.0 = fully on dark line, 0.5 = correctly straddling the edge (setpoint),
    #   1.0 = fully on light background
    #
    # Story: robot is straddling the edge nicely, drifts slightly toward the light
    # side (small, real correction needed), then hits a GAP -- which saturates the
    # sensor to raw=100 (x_t=1.0), the SAME reading a large real drift would give --
    # then the tape resumes and the reading returns to the same mild drift as before
    # the gap.
    raw_xs  = [55, 60, 100, 100, 60]
    xs      = [raw / 100.0 for raw in raw_xs]
    # Target motor correction: keep applying the same *mild* correction through the
    # gap, rather than the *large* correction a saturated 1.0 reading would suggest
    # in isolation -- because the true state of the world (mild drift, tape just
    # missing) hasn't actually changed.
    targets = [0.55, 0.60, 0.58, 0.56, 0.60]

    cell = ScalarLSTMCell(lr=0.3)

    print("=== forward pass BEFORE training (random weights) ===")
    _, caches = cell.forward_sequence(xs)
    for t, c in enumerate(caches):
        print(f"t={t}  raw={raw_xs[t]:3d}  x={c['x_t']:.2f}  f={c['f_t']:.3f}  i={c['i_t']:.3f}  "
              f"g={c['g_t']:+.3f}  o={c['o_t']:.3f}  c_t={c['c_t']:+.3f}  "
              f"h_t={c['h_t']:+.3f}  y_t={c['y_t']:+.3f}  target={targets[t]:.2f}")

    print("\n=== training with truncated BPTT over this 5-step window ===")
    for epoch in range(2000):
        _, caches = cell.forward_sequence(xs)
        loss, grads = cell.backward_sequence(caches, targets)
        cell.update(grads)
        if epoch % 250 == 0:
            print(f"epoch {epoch:4d}   total loss = {loss:.5f}")

    print("\n=== forward pass AFTER training ===")
    _, caches = cell.forward_sequence(xs)
    for t, c in enumerate(caches):
        print(f"t={t}  raw={raw_xs[t]:3d}  x={c['x_t']:.2f}  f={c['f_t']:.3f}  c_t={c['c_t']:+.3f}  "
              f"h_t={c['h_t']:+.3f}  y_t={c['y_t']:+.3f}  target={targets[t]:.2f}")

    print("\nCheck the forget gate f_t at t=2,3 (the gap steps, raw=100 -> x=1.00 saturated):")
    print("values pulled toward 1 mean the cell state -- and the memory that this")
    print("was a MILD drift, not a large one -- survives the saturated reading")
    print("instead of being overwritten by it.")

    