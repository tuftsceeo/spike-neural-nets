"""
Fit a piecewise-linear approximation to 20 points sampled from a sine wave.

The model is a sum of 20 ReLU-built "tent" functions, one centered at each
x-location:

    tent_i(x) = max(0, 1 - |x - knot_i| / spacing)
    y_hat(x)  = sum_i weight_i * tent_i(x)

Each tent is 1 at its own knot and fades linearly to 0 at its neighbors, so
weight_i controls the curve's height near knot_i without affecting the rest
of the curve. That locality is exactly what a nonlinear term buys you: a
plain linear model (w*x + b) can only ever draw one global straight line, no
matter how many weights it has. The tents are what let 20 independent
weights bend the curve at 20 different places, and this is a genuine
function of x -- evaluable anywhere, not just at the training points.

Gradient descent ("backprop") nudges every weight toward its target sine
value, starting from a deliberately bad random guess and animating until it
converges.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

rng = np.random.default_rng()

n_points = 20
knots = np.linspace(0, 4 * np.pi, n_points)
y = np.sin(knots)
spacing = knots[1] - knots[0]


def tent_features(x_eval):
    """Shape (len(x_eval), n_points): tent_i(x) in each column."""
    dist = np.abs(x_eval[:, None] - knots[None, :]) / spacing
    return np.maximum(0.0, 1.0 - dist)


# At the knots themselves, tent_i(knot_j) is 1 if i == j and 0 otherwise, so
# this is just the identity matrix -- each weight maps directly onto its own
# point, exactly as it should for an interpolating fit.
Phi_train = tent_features(knots)

# Deliberately bad starting weights (true sine values only range -1..1).
weights = rng.uniform(-3, 3, size=n_points)

lr = 0.3
tol = 1e-6
max_steps = 100_000
record_every = 1

history = [weights.copy()]
losses = [np.mean((Phi_train @ weights - y) ** 2)]

step = 0
while losses[-1] > tol and step < max_steps:
    # Forward pass through the tent-function network.
    y_hat = Phi_train @ weights

    # Backprop: chain rule through the (fixed) tent features to the weights.
    error = y_hat - y
    grad = Phi_train.T @ (2 * error / n_points)

    weights = weights - lr * grad
    step += 1

    if step % record_every == 0:
        history.append(weights.copy())
        losses.append(np.mean((Phi_train @ weights - y) ** 2))

# Hold the first and last frame for a beat.
hold_start_frames = 15
hold_end_frames = 30
frame_indices = [0] * hold_start_frames + list(range(len(history))) + [len(history) - 1] * hold_end_frames

fig, (ax_curve, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

x_fine = np.linspace(knots.min(), knots.max(), 400)
Phi_fine = tent_features(x_fine)

ax_curve.plot(x_fine, np.sin(x_fine), color="lightgray", lw=1.5, linestyle="--", label="true sine")
ax_curve.scatter(knots, y, color="crimson", zorder=5, s=25, label="target points")
(curve_plot,) = ax_curve.plot([], [], color="steelblue", lw=2, label="network output")
ax_curve.set_xlim(knots.min() - 0.5, knots.max() + 0.5)
ax_curve.set_ylim(min(y.min(), weights.min()) - 1, max(y.max(), weights.max()) + 1)
ax_curve.set_title("Sum of ReLU tent functions fitting a sine wave")
ax_curve.legend(loc="upper right")
title_text = ax_curve.text(0.02, 0.02, "", transform=ax_curve.transAxes, va="bottom", ha="left")

ax_loss.set_xlim(0, len(losses) * record_every)
ax_loss.set_ylim(0, max(losses) * 1.05)
ax_loss.set_xlabel("training step")
ax_loss.set_ylabel("loss (MSE)")
ax_loss.set_title("Loss over training")
(loss_plot,) = ax_loss.plot([], [], color="darkorange", lw=2)


def update(i):
    frame = frame_indices[i]
    w = history[frame]
    y_line = Phi_fine @ w
    curve_plot.set_data(x_fine, y_line)
    step_num = frame * record_every
    title_text.set_text(f"step {step_num}\nloss={losses[frame]:.5f}")

    loss_plot.set_data([j * record_every for j in range(frame + 1)], losses[: frame + 1])
    return curve_plot, loss_plot, title_text


anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=30, blit=False, repeat=False)

plt.tight_layout()
plt.show()
