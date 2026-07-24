"""
A single neuron y = w*x + b learns to pass through two fixed points using
gradient descent ("backprop"), animating the line and the loss as it trains.

Everything is plain scalars (no vectors/matrices) to keep the one-neuron
math explicit. Training starts from a deliberately bad guess and stops as
soon as it converges.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

rng = np.random.default_rng()

# Two target points.
x1, y1 = -2.0, 1.0
x2, y2 = 3.0, 4.0

# The unique line through both points, just so we can force a bad start.
w_true = (y2 - y1) / (x2 - x1)
b_true = y1 - w_true * x1

# Random starting slope/offset, redrawn until it's clearly far from correct.
while True:
    w = rng.uniform(-6, 6)
    b = rng.uniform(-6, 6)
    if abs(w - w_true) + abs(b - b_true) > 6:
        break

lr = 0.01
tol = 1e-6
max_steps = 200_000
record_every = 2

history = [(w, b)]
losses = [((w * x1 + b - y1) ** 2 + (w * x2 + b - y2) ** 2) / 2]

step = 0
while losses[-1] > tol and step < max_steps:
    # Forward pass: one neuron, no activation.
    yhat1 = w * x1 + b
    yhat2 = w * x2 + b

    err1 = yhat1 - y1
    err2 = yhat2 - y2

    # Backprop: chain rule through the neuron for each point, summed.
    grad_w = err1 * x1 + err2 * x2
    grad_b = err1 + err2

    w -= lr * grad_w
    b -= lr * grad_b
    step += 1

    if step % record_every == 0:
        loss = ((w * x1 + b - y1) ** 2 + (w * x2 + b - y2) ** 2) / 2
        history.append((w, b))
        losses.append(loss)

# Hold on the first and last frame for a beat so the start and the
# converged result are both easy to actually look at.
hold_start_frames = 20
hold_end_frames = 40

fig, (ax_line, ax_loss) = plt.subplots(1, 2, figsize=(11, 5))

x_line = np.linspace(min(x1, x2) - 3, max(x1, x2) + 3, 200)

ax_line.scatter([x1, x2], [y1, y2], color="crimson", zorder=5, label="target points")
(line_plot,) = ax_line.plot([], [], color="steelblue", lw=2, label="w x + b")
ax_line.set_xlim(x_line.min(), x_line.max())
ax_line.set_ylim(min(y1, y2, w * x1 + b, w * x2 + b) - 2, max(y1, y2, w * x1 + b, w * x2 + b) + 2)
ax_line.set_title("A single neuron learning to fit two points")
ax_line.legend(loc="upper left")
title_text = ax_line.text(0.98, 0.05, "", transform=ax_line.transAxes, va="bottom", ha="right")

ax_loss.set_xlim(0, len(losses) * record_every)
ax_loss.set_ylim(0, max(losses) * 1.05)
ax_loss.set_xlabel("training step")
ax_loss.set_ylabel("loss (MSE)")
ax_loss.set_title("Loss over training")
(loss_plot,) = ax_loss.plot([], [], color="darkorange", lw=2)

# Frame sequence: hold on frame 0, play through training, hold on the last frame.
frame_indices = [0] * hold_start_frames + list(range(len(history))) + [len(history) - 1] * hold_end_frames


def update(i):
    frame = frame_indices[i]
    w_i, b_i = history[frame]
    y_line = w_i * x_line + b_i
    line_plot.set_data(x_line, y_line)
    step_num = frame * record_every
    title_text.set_text(f"step {step_num}\nw={w_i:.3f}, b={b_i:.3f}\nloss={losses[frame]:.4f}")

    loss_plot.set_data([j * record_every for j in range(frame + 1)], losses[: frame + 1])
    return line_plot, loss_plot, title_text


anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=30, blit=False, repeat=False)

plt.tight_layout()
plt.show()
