"""Topology (add/remove layer), forward propagation, and the backward
"reveal plan" that the step machine walks through one layer at a time.

The network is a strict chain of single-neuron layers: a0 = x (the raw
input), and for layer i (1-indexed): z_i = w_i * a_{i-1} + b_i,
a_i = act_i(z_i). The final layer's activation is the prediction.

Chain rule convention used throughout (and shown in the UI), split into the
two stages the step machine reveals separately -- first the activation
node, then the linear/weight node:
    dL/dz_i = dL/da_i * da_i/dz_i     (revealed at the activation node)
    dL/dw_i = dL/dz_i * dz_i/dw_i     (revealed at the linear/weight node)
where dL/da_i ("grad_out") is the gradient flowing in from the layer
immediately ahead of it (i+1, or the loss itself if i is the last layer),
da_i/dz_i ("act_deriv") is the activation function's own local slope, and
dz_i/dw_i is simply the value that fed this layer (a_{i-1}).
"""
import random

import state
from activations import apply_activation, apply_activation_derivative


def _clip_grad(value: float) -> float:
    limit = state.MAX_GRAD_NORM
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


# ── Topology ──────────────────────────────────────────────────────────────

def add_layer():
    w = random.uniform(*state.INIT_WEIGHT_RANGE)
    b = random.uniform(*state.INIT_WEIGHT_RANGE) if state.biases_enabled else 0.0
    state.layers.append({
        "id": state.next_layer_id(), "w": w, "b": b, "act": "none",
        "last_delta_w": None, "last_delta_b": None,
    })
    reset_training()


def remove_layer(lid: int):
    idx = next((i for i, l in enumerate(state.layers) if l["id"] == lid), None)
    if idx is None:
        return
    state.layers.pop(idx)
    reset_training()


def set_layer_activation(lid: int, act: str):
    layer = next((l for l in state.layers if l["id"] == lid), None)
    if layer is None:
        return
    layer["act"] = act
    reset_training()


def set_biases_enabled(enabled: bool):
    state.biases_enabled = enabled
    for l in state.layers:
        if not enabled:
            l["b"] = 0.0
        l["last_delta_b"] = None
    reset_training()


def randomize_weights():
    for l in state.layers:
        l["w"] = random.uniform(*state.INIT_WEIGHT_RANGE)
        l["b"] = random.uniform(*state.INIT_WEIGHT_RANGE) if state.biases_enabled else 0.0
        l["last_delta_w"] = None
        l["last_delta_b"] = None
    reset_training()


def reset_training():
    """Architecture/weight changes invalidate any in-progress epoch reveal
    and make the loss history's earlier scale meaningless, so wipe both."""
    state.epoch = 0
    state.step_index = 0
    state.plan = []
    state.forward_cache = None
    state.loss_history = []
    state.history = []


# ── Forward propagation ──────────────────────────────────────────────────

def forward_point(x: float) -> tuple[list[float], list[float]]:
    """Returns (pre, post) each length len(layers)+1, index 0 = the raw
    input (pre[0] is unused/None-ish, post[0] = x)."""
    pre = [0.0]
    post = [x]
    a_prev = x
    for l in state.layers:
        z = l["w"] * a_prev + l["b"]
        a = apply_activation(z, l["act"])
        pre.append(z)
        post.append(a)
        a_prev = a
    return pre, post


def predict(x: float) -> float:
    if not state.layers:
        return x
    _, post = forward_point(x)
    return post[-1]


def compute_forward_cache() -> dict:
    """Forward-propagates every dataset point with the CURRENT weights and
    caches everything build_backward_plan() needs, plus the mean loss."""
    points = []
    total_loss = 0.0
    for p in state.dataset:
        pre, post = forward_point(p["x"])
        pred = post[-1]
        err = pred - p["y"]
        total_loss += err ** 2
        points.append({"x": p["x"], "y": p["y"], "pre": pre, "post": post, "pred": pred})
    mean_loss = total_loss / len(points) if points else 0.0
    return {"points": points, "mean_loss": mean_loss}


# ── Backward "reveal plan" ────────────────────────────────────────────────

def build_backward_plan() -> list[dict]:
    """Runs full backprop once (batch-averaged) using state.forward_cache,
    and returns one entry per layer ordered from the LAST layer to the
    FIRST -- the order the step machine reveals them in (each layer is
    then shown in two sub-steps: activation, then linear/weight).

    Each entry carries both the exact batch gradients (grad_w/grad_b, used
    to actually update the weight/bias) and the pedagogical per-stage
    averages (avg_grad_out, avg_act_deriv, avg_delta, avg_source) whose
    products approximate them -- exact whenever the dataset has a single
    point, a close approximation otherwise, and always shown alongside the
    formula they illustrate rather than as the number actually used to
    step."""
    points = state.forward_cache["points"]
    n = len(points)
    L = len(state.layers)
    if n == 0 or L == 0:
        return []

    # grad_out_per_point[k] holds dL/da_L for point k, seeded from the loss;
    # as we walk backward it becomes dL/da_i for the layer currently being
    # visited.
    grad_out_per_point = [2.0 * (pt["pred"] - pt["y"]) for pt in points]

    plan = []
    for i in range(L, 0, -1):
        layer = state.layers[i - 1]
        act = layer["act"]

        grad_w_sum = 0.0
        grad_b_sum = 0.0
        grad_out_sum = 0.0
        act_deriv_sum = 0.0
        delta_sum = 0.0
        source_sum = 0.0
        next_grad_out_per_point = [0.0] * n

        for k, pt in enumerate(points):
            pre_i = pt["pre"][i]
            post_i = pt["post"][i]
            source = pt["post"][i - 1]
            act_deriv = apply_activation_derivative(pre_i, post_i, act)
            grad_out = grad_out_per_point[k]

            delta = grad_out * act_deriv
            grad_w_sum += delta * source
            grad_b_sum += delta

            grad_out_sum += grad_out
            act_deriv_sum += act_deriv
            delta_sum += delta
            source_sum += source

            next_grad_out_per_point[k] = delta * layer["w"]

        grad_w = _clip_grad(grad_w_sum / n)
        grad_b = _clip_grad(grad_b_sum / n)
        avg_grad_out = grad_out_sum / n
        avg_act_deriv = act_deriv_sum / n
        avg_delta = delta_sum / n
        avg_source = source_sum / n

        w_old, b_old = layer["w"], layer["b"]
        w_new = w_old - state.lr * grad_w
        b_new = b_old - state.lr * grad_b if state.biases_enabled else 0.0

        plan.append({
            "layer_id": layer["id"],
            "layer_pos": i,          # 1-indexed position in the chain
            "is_last": i == L,
            "act": act,
            "grad_w": grad_w,
            "grad_b": grad_b,
            "avg_grad_out": avg_grad_out,     # dL/da_i
            "avg_act_deriv": avg_act_deriv,   # da_i/dz_i
            "avg_delta": avg_delta,           # dL/dz_i (exact mean, not a product-of-means approx)
            "avg_source": avg_source,         # dz_i/dw_i (the value that fed this layer)
            "w_old": w_old, "w_new": w_new,
            "b_old": b_old, "b_new": b_new,
        })

        grad_out_per_point = next_grad_out_per_point

    return plan


def apply_plan_step(step_num: int) -> dict:
    """Applies plan[step_num - 1] (step_num is 1..len(plan)) to the real
    layer weights and returns that plan entry. Also records how much each
    value just moved, so the diagram can draw a direction/magnitude
    indicator next to it."""
    entry = state.plan[step_num - 1]
    layer = next(l for l in state.layers if l["id"] == entry["layer_id"])
    layer["last_delta_w"] = entry["w_new"] - entry["w_old"]
    layer["w"] = entry["w_new"]
    if state.biases_enabled:
        layer["last_delta_b"] = entry["b_new"] - entry["b_old"]
        layer["b"] = entry["b_new"]
    return entry


# ── Snapshotting for backward step/epoch ──────────────────────────────────

def take_snapshot():
    # loss_history is stored by LENGTH, not a copied list -- Play can take
    # thousands of snapshots per session, and re-copying the whole
    # (monotonically growing) loss history on every single one of them
    # made each snapshot progressively more expensive the longer Play ran.
    state.history.append({
        "epoch": state.epoch,
        "step_index": state.step_index,
        "layers": [dict(l) for l in state.layers],
        "plan": [dict(p) for p in state.plan],
        "forward_cache": state.forward_cache,
        "loss_history_len": len(state.loss_history),
    })


def restore_snapshot(snap: dict):
    state.epoch = snap["epoch"]
    state.step_index = snap["step_index"]
    for l, saved in zip(state.layers, snap["layers"]):
        l.update(saved)
    state.plan = [dict(p) for p in snap["plan"]]
    state.forward_cache = snap["forward_cache"]
    state.loss_history = state.loss_history[:snap["loss_history_len"]]


# ── Dataset mutation ──────────────────────────────────────────────────────

def add_data_point(x: float, y: float):
    state.dataset.append({"id": state.next_data_id(), "x": x, "y": y})
    reset_training()


def remove_data_point(pid: int):
    idx = next((i for i, p in enumerate(state.dataset) if p["id"] == pid), None)
    if idx is None:
        return
    state.dataset.pop(idx)
    reset_training()
