"""
Why hidden layers matter: "sprint on the line, brake off it"

Same track, same sensor (20 / 40 / 60), but now the target behavior is
NON-MONOTONIC in the sensor reading -- fast when centered, slow when off
either side. A linear net physically cannot do this. A tiny MLP
(1 input -> 3 tanh hidden -> 2 linear outputs) can.

Pure Python, no numpy -- portable to the LEGO hub's MicroPython.
"""

import math
import random

# ---- Training data --------------------------------------------------------
# x = (reading - 40) / 20, targets scaled to 0..1 (multiply by 100 to deploy)
DATA = [
    # (x,   target_L, target_R)
    (-1.0, 0.80, 0.20),   # left of line  -> brake, turn right
    ( 0.0, 0.90, 0.90),   # ON the line   -> full sprint  <- the "bump"
    (+1.0, 0.20, 0.80),   # right of line -> brake, turn left
]

random.seed(3)
LR = 0.1
EPOCHS = 3000
N_HIDDEN = 3


# ============================================================
# Network 1: the old linear net (will FAIL on this data)
# ============================================================
def train_linear():
    w_L, b_L = random.uniform(-1, 1), random.uniform(-1, 1)
    w_R, b_R = random.uniform(-1, 1), random.uniform(-1, 1)
    for epoch in range(EPOCHS):
        loss = 0.0
        for x, t_L, t_R in DATA:
            y_L, y_R = w_L * x + b_L, w_R * x + b_R
            e_L, e_R = t_L - y_L, t_R - y_R
            loss += 0.5 * (e_L**2 + e_R**2)
            w_L += LR * e_L * x;  b_L += LR * e_L
            w_R += LR * e_R * x;  b_R += LR * e_R
        if epoch % 500 == 0 or epoch == EPOCHS - 1:
            print("  epoch {:4d}   loss = {:.6f}".format(epoch, loss))
    predict = lambda x: (w_L * x + b_L, w_R * x + b_R)
    return predict


# ============================================================
# Network 2: 1 -> 3 tanh hidden -> 2 linear outputs
# ============================================================
def train_mlp():
    # hidden layer: h_j = tanh(w[j] * x + b[j])
    w = [random.uniform(-1, 1) for _ in range(N_HIDDEN)]
    b = [random.uniform(-1, 1) for _ in range(N_HIDDEN)]
    # output layer: y_k = sum_j v[k][j] * h_j + c[k]     (k = 0:L, 1:R)
    v = [[random.uniform(-1, 1) for _ in range(N_HIDDEN)] for _ in range(2)]
    c = [random.uniform(-1, 1) for _ in range(2)]

    def forward(x):
        h = [math.tanh(w[j] * x + b[j]) for j in range(N_HIDDEN)]
        y = [sum(v[k][j] * h[j] for j in range(N_HIDDEN)) + c[k]
             for k in range(2)]
        return h, y

    for epoch in range(EPOCHS):
        loss = 0.0
        for x, t_L, t_R in DATA:
            # ---------- forward pass ----------
            h, y = forward(x)
            targets = (t_L, t_R)

            # ---------- backward pass ----------
            # 1. Output-layer deltas (same as the no-hidden-layer version)
            d_out = [targets[k] - y[k] for k in range(2)]
            loss += 0.5 * sum(d * d for d in d_out)

            # 2. THE NEW STEP -- propagate error back through the layer:
            #    each hidden neuron collects blame from BOTH motors,
            #    scaled by how much it fed each one (v[k][j]),
            #    times the tanh derivative (1 - h^2).
            d_hid = [(d_out[0] * v[0][j] + d_out[1] * v[1][j])
                     * (1.0 - h[j] ** 2)
                     for j in range(N_HIDDEN)]

            # 3. Updates (note: use h values from BEFORE changing v)
            for k in range(2):
                for j in range(N_HIDDEN):
                    v[k][j] += LR * d_out[k] * h[j]
                c[k] += LR * d_out[k]
            for j in range(N_HIDDEN):
                w[j] += LR * d_hid[j] * x
                b[j] += LR * d_hid[j]

        if epoch % 500 == 0 or epoch == EPOCHS - 1:
            print("  epoch {:4d}   loss = {:.6f}".format(epoch, loss))

    predict = lambda x: forward(x)[1]
    return predict


def behavior_table(name, predict):
    print("\n  {} behavior (motor speeds 0-100):".format(name))
    for reading in (20, 40, 60):
        x = (reading - 40) / 20
        out = predict(x)
        y_L, y_R = out[0], out[1]
        print("    sensor={:2d}  ->  L {:5.1f}   R {:5.1f}".format(
            reading, 100 * y_L, 100 * y_R))


if __name__ == "__main__":
    print("Targets: sensor 20 -> L80/R20,  sensor 40 -> L90/R90 (sprint!),"
          "  sensor 60 -> L20/R80\n")

    print("LINEAR NET (no hidden layer):")
    lin = train_linear()
    behavior_table("Linear net", lin)
    print("\n  ^ Stuck: loss plateaus at a nonzero floor. It cruises at ~63"
          "\n    on the line instead of 90, because a line can't make a bump.\n")

    print("MLP (1 -> 3 tanh -> 2):")
    mlp = train_mlp()
    behavior_table("MLP", mlp)
    print("\n  ^ The hidden layer builds the bump: sprint at 40, brake at 20/60.")
