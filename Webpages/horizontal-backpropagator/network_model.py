"""Topology (add/remove layer), forward propagation, and the backward
"reveal plan" that the step machine walks through one layer at a time.

The network is a strict chain of single-neuron layers: a0 = x (the raw
input), and for layer i (1-indexed): n_i = w_i * a_{i-1} + b_i,
a_i = act_i(n_i). The final layer's activation is the prediction (shown
in the diagram as ŷ, with the loss L computed from ŷ in its own node).

Chain rule convention used throughout (and shown in the UI), walked
backward one BOX at a time -- an activation node then a linear/weight
node, per layer, plus one boundary hop from the loss into ŷ:
    dL/dŷ = 2(ŷ - y)                  (the boundary hop, from L into ŷ)
    dL/da_i = dL/dn_{i+1} * dn_{i+1}/da_i   (revealed at the activation node;
                                              dL/dŷ * dŷ/da_i for the last layer,
                                              since ŷ IS a_L with no transform)
    dL/dn_i = dL/da_i * da_i/dn_i      (revealed at the linear/weight node)
    dL/dw_i = dL/dn_i * dn_i/dw_i      (shown alongside dL/dn_i, same node)
Note this is the OPPOSITE pairing from a naive reading of the diagram: the
activation node's own local slope (da_i/dn_i) is revealed when the arrow
ARRIVES at the linear node, not when it arrives at the activation node --
because da_i/dn_i measures how THIS activation box turns n_i into a_i, and
a_i is only fully known once the arrow has already passed through it.
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
    input (pre[0] is unused/None-ish, post[0] = x). pre[i] is n_i (the
    pre-activation sum), post[i] is a_i."""
    pre = [0.0]
    post = [x]
    a_prev = x
    for l in state.layers:
        n = l["w"] * a_prev + l["b"]
        a = apply_activation(n, l["act"])
        pre.append(n)
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

    The UI only ever shows the SYMBOLIC chain-rule formula, never a
    plugged-in number, so this only needs to carry what the actual
    weight/bias update requires -- not any of the intermediate averages
    that used to exist purely to populate a "= a * b ≈ c" numeric line."""
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
        next_grad_out_per_point = [0.0] * n

        for k, pt in enumerate(points):
            pre_i = pt["pre"][i]
            post_i = pt["post"][i]
            source = pt["post"][i - 1]
            act_deriv = apply_activation_derivative(pre_i, post_i, act)
            grad_out = grad_out_per_point[k]

            delta = grad_out * act_deriv       # dL/dn_i for this point
            grad_w_sum += delta * source
            grad_b_sum += delta

            next_grad_out_per_point[k] = delta * layer["w"]

        grad_w = _clip_grad(grad_w_sum / n)
        grad_b = _clip_grad(grad_b_sum / n)

        w_old, b_old = layer["w"], layer["b"]
        w_new = w_old - state.lr * grad_w
        b_new = b_old - state.lr * grad_b if state.biases_enabled else 0.0

        plan.append({
            "layer_id": layer["id"],
            "layer_pos": i,          # 1-indexed position in the chain
            "is_last": i == L,
            "grad_w": grad_w,
            "grad_b": grad_b,
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
