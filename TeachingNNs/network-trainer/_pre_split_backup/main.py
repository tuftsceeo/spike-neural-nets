"""
Neural Network Trainer — main.py
PyScript 2026.3.1

Structure: independent columns, chained through any number of layers.
  Inputs  -- each with its own device/channel + plot
  Layers  -- each layer has its own column of neurons AND its own
             activation block (function + optional custom piecewise editor).
             layers[0]'s neurons take a weighted sum of the raw inputs;
             every later layer's neurons take a weighted sum of the
             *previous* layer's activated neuron outputs.
             "+ Layer" (between the last activation block and the outputs)
             appends a brand new neuron layer + activation block.
  Outputs -- each with its own device/channel + a scatter plot of output
             vs. input. Outputs pair up with the LAST layer's neurons
             positionally (1st output <-> 1st neuron of the last layer,
             etc). Any neuron in the last layer beyond the current number
             of outputs is shown greyed out until a matching output exists.
  A column (inputs / a layer's neurons / outputs) can never be emptied --
  the last remaining item's delete button is hidden.

Training loop (manual, no hardware involved):
  Randomize weights -- gives every neuron a fresh random weight vector + bias.
  Test              -- forward-propagates the CURRENT input reading (without
                        touching the hardware outputs) and drops a point on
                        each output's scatter plot at (first input's value,
                        predicted output).
  Backpropagate      -- compares each output's last Test prediction against
                        the target value typed into that output's "target y"
                        box, and nudges every weight/bias one gradient-descent
                        step to reduce that error.
  Run / Stop         -- unchanged: continuously forward-propagates live
                        sensor input straight into the hardware outputs.
"""
import math
import random
import asyncio
from pyscript import document, window, when
from pyscript.ffi import create_proxy
from Device import Element
import legoeducation as le
import numpy as np

import traceback
import threading
import concurrent.futures

from pyscript.js_modules import Plotly
import plot

# ── State ──────────────────────────────────────────────────────────────────────

inputs:  list[dict] = []   # {id, name, prev_device, prev_channel}
layers:  list[dict] = []   # {id, neurons: [...], act_fn, custom_activation: {expr, pieces}}
outputs: list[dict] = []   # {id, prev_device, prev_channel}

input_counter  = 0
neuron_counter = 0
layer_counter  = 0
output_counter = 0

devices: list[Element] = []
is_running       = False
run_tick         = 0   # ticks since play_network() started, used as the x-axis for the live output plot
all_plots: dict[str, object] = {}

debug_mode = False

# Zoom only affects the network diagram's spacing/box sizes (via the CSS
# --zoom variable) -- never font sizes or arrow stroke widths.
# Zoom only affects the network diagram's spacing/box sizes (via the CSS
# --zoom variable) -- never font sizes or arrow stroke widths, UNTIL zoom
# drops below TEXT_ZOOM_THRESHOLD, at which point text also starts
# shrinking (via --text-zoom) so the diagram can get extra compact.
ZOOM_MIN  = 0.2
ZOOM_MAX  = 1.0
ZOOM_STEP = 0.1
TEXT_ZOOM_THRESHOLD = 0.5
zoom_level = 1.0
input_values:       dict[int, float] = {}   # iid -> raw input reading
neuron_pre_values:  dict[int, float] = {}   # nid -> weighted sum + bias
neuron_post_values: dict[int, float] = {}   # nid -> post-activation value
output_values:      dict[int, float] = {}   # oid -> value routed to that output

# Training (manual test / backpropagate loop) --------------------------------
DEFAULT_LEARNING_RATE = 0.05   # seeds the on-page learning-rate box; the box wins after that
# layer_sources_snapshot[i] = the list of source values (raw inputs for layer 0,
# previous layer's post-activations otherwise) that fed layers[i] during the
# most recent Test() call. Backpropagate() needs this to compute weight
# gradients, since forward() overwrites neuron_pre/post_values continuously
# while the network is Running.
# Dataset-based training (Phase 1+) -------------------------------------------
training_data: list[dict] = []   # [{"id": int, "xs": {iid: float}, "ys": {oid: float}}, ...]
data_point_counter = 0
loss_history: list[float] = []
normalize_enabled = True

ACTIVATION_OPTIONS = [
    ("None",     ""),
    ("ReLU",     "relu"),
    ("Sigmoid",  "sigmoid"),
    ("Tanh",     "tanh"),
    ("Softplus", "softplus"),
    ("Custom",   "custom"),
]

ARROW_COLOR = "#1e40af"

piece_counter = 0   # global, so piece DOM ids never collide across layers

CUSTOM_ACT_NAMES = {
    "abs": abs, "max": max, "min": min, "round": round,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e, "inf": math.inf,
}

import re

IMPLICIT_MULT_RE = re.compile(r'(?<=[0-9])(?=[a-zA-Z(])|(?<=[a-zA-Z)])(?=[0-9(])')

def normalize_expr(expr: str) -> str:
    cleaned = expr.replace("^", "**")
    cleaned = IMPLICIT_MULT_RE.sub("*", cleaned)
    return cleaned

def safe_eval_expr(expr: str, x: float) -> float:
    if not expr or not expr.strip():
        return x
    cleaned = normalize_expr(expr)
    ns = dict(CUSTOM_ACT_NAMES)
    ns["x"] = x
    try:
        return float(eval(cleaned, {"__builtins__": {}}, ns))
    except Exception:
        print("Custom activation eval error:\n" + traceback.format_exc())
        return x

def parse_bound(raw: str):
    if raw is None:
        return None
    s = raw.strip().lower().replace("∞", "inf")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def apply_custom_activation(x: float, custom_activation: dict) -> float:
    pieces = custom_activation["pieces"]
    if not pieces:
        return safe_eval_expr(custom_activation["expr"], x)
    for p in pieces:
        lo, hi = parse_bound(p["lo"]), parse_bound(p["hi"])

        if lo is None:
            lo_ok = True
        elif p["lo_op"] == "<":
            lo_ok = lo < x
        elif p["lo_op"] == "<=":
            lo_ok = lo <= x
        elif p["lo_op"] == ">":
            lo_ok = lo > x
        elif p["lo_op"] == ">=":
            lo_ok = lo >= x
        else:
            lo_ok = True

        if hi is None:
            hi_ok = True
        elif p["hi_op"] == "<":
            hi_ok = x < hi
        elif p["hi_op"] == "<=":
            hi_ok = x <= hi
        elif p["hi_op"] == ">":
            hi_ok = x > hi
        elif p["hi_op"] == ">=":
            hi_ok = x >= hi
        else:
            hi_ok = True

        if lo_ok and hi_ok:
            return safe_eval_expr(p["expr"], x)
    return 0.0

# ── Helpers ────────────────────────────────────────────────────────────────────
def sigmoid_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))
    
def apply_activation(x: float, fn: str, custom_activation: dict | None = None) -> float:
    if fn == "relu":
        return max(0.0, x)
    elif fn == "sigmoid":
        return sigmoid_numpy(x)
    elif fn == "tanh":
        return math.tanh(x)
    elif fn == "softplus":
        return math.log(1.0 + math.exp(x))
    elif fn == "custom":
        return apply_custom_activation(x, custom_activation or {"expr": "x", "pieces": []})
    else:
        return x

def apply_activation_derivative(pre_act: float, post_act: float, fn: str,
                                 custom_activation: dict | None = None) -> float:
    """d(post_act)/d(pre_act), used by backpropagate() for the chain rule."""
    if fn == "relu":
        return 1.0 if pre_act > 0 else 0.0
    elif fn == "sigmoid":
        return post_act * (1.0 - post_act)
    elif fn == "tanh":
        return 1.0 - post_act ** 2
    elif fn == "softplus":
        return sigmoid_numpy(pre_act)
    elif fn == "custom":
        h = 1e-4
        ca = custom_activation or {"expr": "x", "pieces": []}
        return (apply_custom_activation(pre_act + h, ca) - apply_custom_activation(pre_act - h, ca)) / (2 * h)
    else:
        return 1.0

def device_by_name(name: str) -> Element | None:
    for d in devices:
        if d.name == name:
            return d
    return None

def get_id(id_: str):
    return document.getElementById(id_)

def input_by_id(iid: int) -> dict | None:
    return next((i for i in inputs if i["id"] == iid), None)

def output_by_id(oid: int) -> dict | None:
    return next((o for o in outputs if o["id"] == oid), None)

def layer_by_id(lid: int) -> dict | None:
    return next((l for l in layers if l["id"] == lid), None)

def layer_index_by_id(lid: int) -> int | None:
    for i, l in enumerate(layers):
        if l["id"] == lid:
            return i
    return None

def neuron_by_id(nid: int) -> dict | None:
    for l in layers:
        for n in l["neurons"]:
            if n["id"] == nid:
                return n
    return None

def get_source_count(layer_idx: int) -> int:
    """How many weight slots a neuron in this layer needs."""
    if layer_idx == 0:
        return len(inputs)
    return len(layers[layer_idx - 1]["neurons"])

def get_source_labels(layer_idx: int) -> list[str]:
    """Display labels for each weight term in this layer's equations."""
    if layer_idx == 0:
        return [inp["name"] for inp in inputs]
    return [f"n{idx + 1}" for idx in range(len(layers[layer_idx - 1]["neurons"]))]

def get_device_options_html() -> str:
    opts = '<option class="dev-dropdown" value="">— device —</option>'
    for device in devices:
        opts += f'<option value="{device.name}">{device.name}</option>'
    return opts

def get_in_channels_html(device: Element | None = None) -> str:
    if device is None:
        return '<option value="">— value —</option>'
    try:
        state = device.state
        return "".join(f'<option value="{key}">{key}</option>' for key in state.keys())
    except Exception:
        return '<option value="">— value —</option>'

def get_out_channels_html(device: Element | None = None) -> str:
    if device is None:
        return '<option value="">— value —</option>'
    try:
        opts = device.get_out_list()
        return "".join(f'<option value="{opt}">{opt}</option>' for opt in opts)
    except Exception:
        return '<option value="">— value —</option>'

def get_learning_rate() -> float:
    el = get_id("learning-rate-input")
    try:
        val = float(el.value) if el else DEFAULT_LEARNING_RATE
        return val if val > 0 else DEFAULT_LEARNING_RATE
    except (ValueError, TypeError):
        return DEFAULT_LEARNING_RATE

def populate_layer_act_select(lid: int):
    sel = get_id(f"act-select-{lid}")
    if not sel:
        return
    html = ""
    for label, val in ACTIVATION_OPTIONS:
        html += f'<option value="{val}">{label}</option>'
    sel.innerHTML = html

# ── SVG arrow primitives ───────────────────────────────────────────────────────

def svg_path(d: str, stroke_w: float = 2.0):
    p = document.createElementNS("http://www.w3.org/2000/svg", "path")
    p.setAttribute("d", d)
    p.setAttribute("fill", "none")
    p.setAttribute("stroke", ARROW_COLOR)
    p.setAttribute("stroke-width", str(stroke_w))
    p.setAttribute("stroke-linecap", "round")
    p.setAttribute("stroke-linejoin", "round")
    return p

def arrowhead(svg_el, x2: float, y2: float, dx: float, dy: float,
              stroke_w: float = 2.0, size: float = 9.0):
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-9:
        return
    ux, uy = dx / mag, dy / mag
    spread = math.radians(26)
    cs, sn = math.cos(spread), math.sin(spread)
    wx1, wy1 =  ux * cs - uy * sn,  ux * sn + uy * cs
    wx2, wy2 =  ux * cs + uy * sn, -ux * sn + uy * cs
    ax1, ay1 = x2 - size * wx1, y2 - size * wy1
    ax2, ay2 = x2 - size * wx2, y2 - size * wy2
    d = f"M{x2:.2f},{y2:.2f} L{ax1:.2f},{ay1:.2f} M{x2:.2f},{y2:.2f} L{ax2:.2f},{ay2:.2f}"
    svg_el.appendChild(svg_path(d, stroke_w))

def debug_label(svg_el, x: float, y: float, value: float):
    text = f"{value:.2f}"
    char_w, pad_x, h = 6.2, 6, 18
    w = len(text) * char_w + pad_x * 2

    rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
    rect.setAttribute("x", str(x - w / 2))
    rect.setAttribute("y", str(y - h / 2))
    rect.setAttribute("width", str(w))
    rect.setAttribute("height", str(h))
    rect.setAttribute("rx", "5")
    rect.setAttribute("fill", "#1a1d2e")
    rect.setAttribute("stroke", "#ffffff")
    rect.setAttribute("stroke-width", "1")

    txt = document.createElementNS("http://www.w3.org/2000/svg", "text")
    txt.setAttribute("x", str(x))
    txt.setAttribute("y", str(y + 3.5))
    txt.setAttribute("text-anchor", "middle")
    txt.setAttribute("font-family", "JetBrains Mono, monospace")
    txt.setAttribute("font-size", "10")
    txt.setAttribute("font-weight", "700")
    txt.setAttribute("fill", "#ffffff")
    txt.textContent = text

    svg_el.appendChild(rect)
    svg_el.appendChild(txt)

def straight_arrow(svg_el, x1, y1, x2, y2, stroke_w: float = 2.0, debug_value=None,
                    label_t: float = 0.5, label_side: float = 1.0):
    d = f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}"
    svg_el.appendChild(svg_path(d, stroke_w))
    arrowhead(svg_el, x2, y2, x2 - x1, y2 - y1, stroke_w)

    if debug_mode and debug_value is not None:
        mx, my = x1 + (x2 - x1) * label_t, y1 + (y2 - y1) * label_t
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        px, py = -dy / length, dx / length
        offset = 13 * label_side
        debug_label(svg_el, mx + px * offset, my + py * offset, debug_value)

def fan_arrows(svg_el, src_pts, tgt_pts, src_heights, tgt_heights, value_fn):
    """Bipartite fan of straight arrows from every src point to every tgt point,
    spread vertically at both ends so lines don't all overlap."""
    n_src, n_tgt = len(src_pts), len(tgt_pts)
    SPREAD_FRAC = 0.55
    MAX_STEP = 14

    for src_idx, (sx, sy, sid) in enumerate(src_pts):
        for tgt_idx, (tx, ty, tid) in enumerate(tgt_pts):
            if n_tgt > 1:
                src_h    = src_heights.get(sid, 120)
                spread_h = min(src_h * SPREAD_FRAC, (n_tgt - 1) * MAX_STEP)
                tail_y   = sy - spread_h / 2 + (tgt_idx / (n_tgt - 1)) * spread_h
            else:
                tail_y = sy

            if n_src > 1:
                tgt_h    = tgt_heights.get(tid, 80)
                spread_h = min(tgt_h * SPREAD_FRAC, (n_src - 1) * MAX_STEP)
                head_y   = ty - spread_h / 2 + (src_idx / (n_src - 1)) * spread_h
            else:
                head_y = ty

            # Fan lines cross each other in the middle of the gap. Stagger each
            # line's debug label along its own length (and alternate which side
            # it sits on) so labels on crossing lines don't land on top of
            # each other.
            t_step   = tgt_idx - (n_tgt - 1) / 2
            label_t  = 0.5 + (t_step / max(n_tgt, 1)) * 0.6
            label_t  = min(max(label_t, 0.15), 0.85)
            label_side = 1.0 if (src_idx + tgt_idx) % 2 == 0 else -1.0

            straight_arrow(svg_el, sx, tail_y, tx, head_y, stroke_w=1.8,
                           debug_value=value_fn(sid) if value_fn else None,
                           label_t=label_t, label_side=label_side)

# ── Main redraw ────────────────────────────────────────────────────────────────

def _anchor_points(id_prefix: str, item_list: list, id_key: str = "id"):
    """Returns (left_pts, right_pts, heights) for a list of items whose DOM
    cell has id f'{id_prefix}{item[id_key]}'."""
    wrap_el = get_id("network-wrap")
    wr = wrap_el.getBoundingClientRect()
    left_pts, right_pts, heights = [], [], {}
    for item in item_list:
        iid = item[id_key]
        el = get_id(f"{id_prefix}{iid}")
        if el:
            r = el.getBoundingClientRect()
            x = r.left - wr.left
            y = r.top - wr.top + r.height / 2
            left_pts.append((x, y, iid))
            right_pts.append((x + r.width, y, iid))
            heights[iid] = r.height
    return left_pts, right_pts, heights

def redraw_arrows():
    svg_el  = get_id("arrow-svg")
    wrap_el = get_id("network-wrap")
    if (not svg_el or not wrap_el or not inputs or not layers
            or not layers[0]["neurons"] or not outputs):
        if svg_el:
            svg_el.innerHTML = ""
        return

    svg_el.innerHTML = ""

    in_l_pts, in_r_pts, in_heights = _anchor_points("cell-input-", inputs)
    out_l_pts, _, _                = _anchor_points("cell-output-", outputs)

    if not in_r_pts or not out_l_pts:
        return

    def rel_box(el):
        wr = wrap_el.getBoundingClientRect()
        r = el.getBoundingClientRect()
        return {"x": r.left - wr.left, "y": r.top - wr.top, "w": r.width, "h": r.height}

    # 1. Input -> first layer's neurons (fan)
    layer0_l_pts, layer0_r_pts, layer0_heights = _anchor_points("cell-neuron-", layers[0]["neurons"])
    if not layer0_l_pts:
        return
    fan_arrows(svg_el, in_r_pts, layer0_l_pts, in_heights, layer0_heights,
               value_fn=lambda iid: input_values.get(iid))

    prev_r_pts, prev_heights = layer0_r_pts, layer0_heights

    for idx, layer in enumerate(layers):
        lid = layer["id"]
        neuron_l_pts, neuron_r_pts, neuron_heights = _anchor_points("cell-neuron-", layer["neurons"])
        if not neuron_l_pts:
            return
        act_el = get_id(f"act-box-{lid}")
        if not act_el:
            continue
        box = rel_box(act_el)

        # neurons_i -> act_i (converge, 1:1 at each neuron's own y)
        for (nx, ny, nid) in neuron_r_pts:
            straight_arrow(svg_el, nx, ny, box["x"], ny, stroke_w=2.0,
                           debug_value=neuron_pre_values.get(nid))

        act_rx = box["x"] + box["w"]

        if idx + 1 < len(layers):
            # act_i -> neurons_{i+1} (fan: every activated neuron feeds every next neuron)
            next_l_pts, _, next_heights = _anchor_points("cell-neuron-", layers[idx + 1]["neurons"])
            src_pts_from_box = [(act_rx, ny, nid) for (nx, ny, nid) in neuron_r_pts]
            fan_arrows(svg_el, src_pts_from_box, next_l_pts, neuron_heights, next_heights,
                       value_fn=lambda nid: neuron_post_values.get(nid))
        else:
            # last layer -> outputs (1:1 diverge, matching positional pairing)
            for (ox, oy, oid) in out_l_pts:
                straight_arrow(svg_el, act_rx, oy, ox, oy, stroke_w=2.0,
                               debug_value=output_values.get(oid))

# ── Item HTML ──────────────────────────────────────────────────────────────────

def delete_x_svg() -> str:
    return ('<svg width="10" height="10" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="3" stroke-linecap="round">'
            '<path d="M18 6L6 18M6 6l12 12"/></svg>')

def make_input_html(inp: dict) -> str:
    iid  = inp["id"]
    name = inp["name"]
    dev_opts = get_device_options_html()
    return f"""
<div class="network-item" id="item-input-{iid}" data-id="{iid}">
    <div class="cell-node" id="cell-input-{iid}">
        <div class="cell-label">
            <input type="text" class="name-input" id="name-input-{iid}"
                   value="{name}" maxlength="12" />
        </div>
        <div class="node-card input-node hover-target">
            <button class="item-delete-btn" id="del-input-{iid}" title="Remove input">{delete_x_svg()}</button>
            <div class="node-header">
                <select class="node-device-select" id="dev-in-{iid}">{dev_opts}</select>
                <span class="node-reading" id="reading-in-{iid}">—</span>
            </div>
            <div class="node-plot">
                <div class="plot-canvas" id="plot-in-{iid}" width="170" height="70"></div>
            </div>
            <div class="plot-footer">
                <select class="channel-select" id="chan-in-{iid}">
                    <option value="">— value —</option>
                </select>
            </div>
        </div>
    </div>
</div>
"""

def make_neuron_eq_inner_html(n: dict, labels: list[str]) -> str:
    nid = n["id"]
    parts = ""
    for i, label in enumerate(labels):
        coeff_val = n["weights"][i] if i < len(n["weights"]) else 1.0
        if i > 0:
            parts += '<span class="eq-op">+</span>'
        parts += (
            f'<input type="number" step="any" value="{coeff_val:.2f}"'
            f' class="eq-num-input" id="coeff-{nid}-{i}" data-neuron="{nid}" data-idx="{i}" />'
            f'<span class="eq-var" id="var-label-{nid}-{i}">{label}</span>'
        )
    parts += (
        f'<span class="eq-op">+</span>'
        f'<input type="number" step="any" value="{n["bias"]:.2f}"'
        f' class="eq-bias-input" id="bias-{nid}" data-neuron="{nid}" />'
    )
    return parts

def make_neuron_html(n: dict, labels: list[str]) -> str:
    nid = n["id"]
    return f"""
<div class="network-item" id="item-neuron-{nid}" data-id="{nid}">
    <div class="cell-node" id="cell-neuron-{nid}">
        <div class="eq-node hover-target" id="eq-node-{nid}">
            <button class="item-delete-btn" id="del-neuron-{nid}" title="Remove neuron">{delete_x_svg()}</button>
            <div class="eq-inline" id="eq-inline-{nid}">
                {make_neuron_eq_inner_html(n, labels)}
            </div>
        </div>
    </div>
</div>
"""

def make_output_html(out: dict) -> str:
    oid = out["id"]
    dev_opts = get_device_options_html()
    return f"""
<div class="network-item" id="item-output-{oid}" data-id="{oid}">
    <div class="cell-node" id="cell-output-{oid}">
        <div class="node-card output-node hover-target">
            <button class="item-delete-btn" id="del-output-{oid}" title="Remove output">{delete_x_svg()}</button>
            <div class="node-header">
                <select class="node-device-select" id="dev-out-{oid}">{dev_opts}</select>
                <span class="node-reading" id="reading-out-{oid}">—</span>
            </div>
            <div class="node-plot">
                <div class="plot-canvas" id="plot-out-{oid}" width="170" height="70"></div>
            </div>
            <div class="plot-footer">
                <select class="channel-select" id="chan-out-{oid}">
                    <option value="">— value —</option>
                </select>
            </div>
        </div>
    </div>
</div>
"""

def make_layer_neurons_col_html(layer: dict) -> str:
    lid = layer["id"]
    return f"""
<div class="col col-neurons" id="col-neurons-{lid}" data-layer="{lid}">
    <div class="col-items" id="neurons-container-{lid}"></div>
    <button class="btn-add-col" id="add-neuron-btn-{lid}" title="Add neuron">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5">
            <path d="M12 5v14M5 12h14"/>
        </svg>
        Neuron
    </button>
</div>
"""

def make_layer_activation_col_html(layer: dict) -> str:
    lid = layer["id"]
    return f"""
<div class="col col-activation" id="act-col-{lid}" data-layer="{lid}">
    <div class="act-box" id="act-box-{lid}">
        <div class="act-box-label">Activation</div>
        <select class="act-select" id="act-select-{lid}"></select>
        <button class="act-help-btn" id="act-help-btn-{lid}" title="What is an activation function?">?</button>

        <div class="custom-act-box hidden" id="custom-act-box-{lid}">
            <div class="custom-main-row">
                <span class="custom-fx-label">f(x) =</span>
                <input type="text" class="custom-eq-input" id="custom-default-expr-{lid}" value="x" />
                <div class="custom-pieces-wrap hidden" id="custom-pieces-wrap-{lid}">
                    <div class="custom-brace">{{</div>
                    <div class="custom-pieces-list" id="custom-pieces-list-{lid}"></div>
                </div>
            </div>
            <button class="btn-add-piece" id="add-piece-btn-{lid}">+ Add piece</button>
        </div>
    </div>
</div>
"""

# ── Sync helpers ───────────────────────────────────────────────────────────────

def sync_var_labels():
    """Refresh the input-name labels shown inside layer 0's equations
    (deeper layers use auto-generated positional labels, unaffected)."""
    if not layers:
        return
    for i, inp in enumerate(inputs):
        for n in layers[0]["neurons"]:
            el = get_id(f"var-label-{n['id']}-{i}")
            if el:
                el.textContent = inp["name"]

def rebuild_layer_eq_html(layer_idx: int):
    """Rebuild every neuron's equation row in this layer to match its
    current source count/labels (input count for layer 0, previous
    layer's neuron count otherwise)."""
    if layer_idx < 0 or layer_idx >= len(layers):
        return
    layer = layers[layer_idx]
    source_count = get_source_count(layer_idx)
    labels = get_source_labels(layer_idx)
    for n in layer["neurons"]:
        nid = n["id"]
        container = get_id(f"eq-inline-{nid}")
        if not container:
            continue
        while len(n["weights"]) < source_count:
            n["weights"].append(1.0)
        if len(n["weights"]) > source_count:
            n["weights"] = n["weights"][:source_count]
        container.innerHTML = make_neuron_eq_inner_html(n, labels)
        bind_neuron_eq_inputs(nid)

def sync_downstream_after_neuron_count_change(layer_idx: int, removed_pos: int | None = None):
    """Call after a neuron was added to or removed from layers[layer_idx].
    Keeps the NEXT layer's weight vectors in sync (append 1.0 on add,
    pop the matching index on remove) and rebuilds its equation display."""
    if layer_idx + 1 >= len(layers):
        return
    next_layer = layers[layer_idx + 1]
    if removed_pos is not None:
        for n in next_layer["neurons"]:
            if removed_pos < len(n["weights"]):
                n["weights"].pop(removed_pos)
    else:
        for n in next_layer["neurons"]:
            n["weights"].append(1.0)
    rebuild_layer_eq_html(layer_idx + 1)

def update_neuron_usage_state():
    """Neurons in the LAST layer beyond the number of outputs aren't wired
    to anything yet -- grey them out until a matching output is added."""
    for li, layer in enumerate(layers):
        is_last = (li == len(layers) - 1)
        for i, n in enumerate(layer["neurons"]):
            eq_el = get_id(f"eq-node-{n['id']}")
            if not eq_el:
                continue
            if is_last and i >= len(outputs):
                eq_el.classList.add("neuron-unused")
            else:
                eq_el.classList.remove("neuron-unused")

def update_delete_visibility(item_list: list, prefix: str):
    """Hide the delete button on every item in a column when only one remains,
    so a column can never be emptied."""
    can_delete = len(item_list) > 1
    for it in item_list:
        btn = get_id(f"del-{prefix}-{it['id']}")
        if btn:
            if can_delete:
                btn.classList.remove("item-delete-hidden")
            else:
                btn.classList.add("item-delete-hidden")

def update_neuron_delete_visibility(layer: dict):
    """The first neuron in a layer can be deleted (removing the whole layer)
    whenever another layer exists, even if it's that layer's only neuron.
    Any other neuron follows the normal 'can't empty this layer' rule."""
    neurons = layer["neurons"]
    layer_removable = len(layers) > 1
    for i, n in enumerate(neurons):
        btn = get_id(f"del-neuron-{n['id']}")
        if not btn:
            continue
        removable = (i == 0 and layer_removable) or len(neurons) > 1
        if removable:
            btn.classList.remove("item-delete-hidden")
        else:
            btn.classList.add("item-delete-hidden")

def refresh_all_neuron_delete_visibility():
    for layer in layers:
        update_neuron_delete_visibility(layer)

def refresh_device_dropdowns():
    dev_opts = get_device_options_html()
    for inp in inputs:
        sel = get_id(f"dev-in-{inp['id']}")
        if sel:
            cur = sel.value
            sel.innerHTML = dev_opts
            sel.value = cur
    for out in outputs:
        sel = get_id(f"dev-out-{out['id']}")
        if sel:
            cur = sel.value
            sel.innerHTML = dev_opts
            sel.value = cur

# ── Event binding: neuron equation ──────────────────────────────────────────────

def bind_neuron_eq_inputs(nid: int):
    n = neuron_by_id(nid)
    if not n:
        return
    for i in range(len(n["weights"])):
        inp_el = get_id(f"coeff-{nid}-{i}")
        if inp_el:
            def make_ch(neuron, idx):
                def h(evt):
                    try:
                        neuron["weights"][idx] = float(evt.target.value)
                    except (ValueError, TypeError):
                        pass
                return create_proxy(h)
            inp_el.addEventListener("input", make_ch(n, i))
    bias_el = get_id(f"bias-{nid}")
    if bias_el:
        def make_bh(neuron):
            def h(evt):
                try:
                    neuron["bias"] = float(evt.target.value)
                except (ValueError, TypeError):
                    pass
            return create_proxy(h)
        bias_el.addEventListener("input", make_bh(n))

# ── Event binding: device/channel plumbing (shared shape for in/out) ───────────

def _attach_plot_to_device(plot_id: str, dev_name: str, channel: str | None):
    plot_obj = all_plots.get(plot_id)
    dev = device_by_name(dev_name)
    if dev and channel and plot_obj:
        dev.plots.append(plot_obj)
        dev.plot_vars.append(channel)

def _detach_plot_from_device(plot_id: str, dev_name: str | None):
    if not dev_name:
        return
    plot_obj = all_plots.get(plot_id)
    dev = device_by_name(dev_name)
    if dev and plot_obj and plot_obj in dev.plots:
        idx = dev.plots.index(plot_obj)
        dev.plots.pop(idx)
        if idx < len(dev.plot_vars):
            dev.plot_vars.pop(idx)

def on_input_device_change(evt):
    sel = evt.target
    iid = int(sel.id[len("dev-in-"):])
    inp = input_by_id(iid)
    if not inp:
        return
    dev_name = sel.value
    chan_sel = get_id(f"chan-in-{iid}")
    matched  = device_by_name(dev_name)
    if chan_sel:
        chan_sel.innerHTML = get_in_channels_html(matched)

    plot_id = f"plot-in-{iid}"
    _detach_plot_from_device(plot_id, inp.get("prev_device"))
    first_channel = None
    if chan_sel and chan_sel.options.length > 0:
        first_channel = chan_sel.options.item(0).value
    _attach_plot_to_device(plot_id, dev_name, first_channel)

    inp["prev_device"]  = dev_name
    inp["prev_channel"] = first_channel

def on_input_channel_change(evt):
    sel = evt.target
    iid = int(sel.id[len("chan-in-"):])
    inp = input_by_id(iid)
    if not inp:
        return
    dev_sel = get_id(f"dev-in-{iid}")
    dev_name = dev_sel.value if dev_sel else ""
    channel  = sel.value
    plot_id  = f"plot-in-{iid}"

    _detach_plot_from_device(plot_id, inp.get("prev_device"))
    _attach_plot_to_device(plot_id, dev_name, channel)

    inp["prev_device"]  = dev_name
    inp["prev_channel"] = channel

def on_output_device_change(evt):
    sel = evt.target
    oid = int(sel.id[len("dev-out-"):])
    out = output_by_id(oid)
    if not out:
        return
    dev_name = sel.value
    chan_sel = get_id(f"chan-out-{oid}")
    matched  = device_by_name(dev_name)
    if chan_sel:
        chan_sel.innerHTML = get_out_channels_html(matched)

    # NOTE: unlike inputs, the output's live plot is fed exclusively by
    # forward()'s per-tick push of the network's computed output (so it
    # shows what the network is commanding, not raw hardware feedback) --
    # it's deliberately never attached to device.plots/plot_vars.
    first_channel = None
    if chan_sel and chan_sel.options.length > 0:
        first_channel = chan_sel.options.item(0).value

    out["prev_device"]  = dev_name
    out["prev_channel"] = first_channel

def on_output_channel_change(evt):
    sel = evt.target
    oid = int(sel.id[len("chan-out-"):])
    out = output_by_id(oid)
    if not out:
        return
    dev_sel = get_id(f"dev-out-{oid}")
    dev_name = dev_sel.value if dev_sel else ""
    channel  = sel.value

    out["prev_device"]  = dev_name
    out["prev_channel"] = channel

# ── Item binding ────────────────────────────────────────────────────────────────

def bind_input_events(iid: int):
    inp = input_by_id(iid)
    if not inp:
        return
    name_el = get_id(f"name-input-{iid}")
    if name_el:
        def h(evt):
            inp["name"] = evt.target.value.strip() or inp["name"]
            sync_var_labels()
        name_el.addEventListener("input", create_proxy(h))

    dev_sel = get_id(f"dev-in-{iid}")
    if dev_sel:
        dev_sel.addEventListener("change", create_proxy(on_input_device_change))

    chan_sel = get_id(f"chan-in-{iid}")
    if chan_sel:
        chan_sel.addEventListener("change", create_proxy(on_input_channel_change))

    del_btn = get_id(f"del-input-{iid}")
    if del_btn:
        del_btn.addEventListener("click", create_proxy(lambda evt: delete_input(iid)))

def bind_neuron_events(lid: int, nid: int):
    n = neuron_by_id(nid)
    if not n:
        return
    bind_neuron_eq_inputs(nid)

    del_btn = get_id(f"del-neuron-{nid}")
    if del_btn:
        del_btn.addEventListener("click", create_proxy(lambda evt: delete_neuron(lid, nid)))

def bind_output_events(oid: int):
    out = output_by_id(oid)
    if not out:
        return
    dev_sel = get_id(f"dev-out-{oid}")
    if dev_sel:
        dev_sel.addEventListener("change", create_proxy(on_output_device_change))

    chan_sel = get_id(f"chan-out-{oid}")
    if chan_sel:
        chan_sel.addEventListener("change", create_proxy(on_output_channel_change))

    del_btn = get_id(f"del-output-{oid}")
    if del_btn:
        del_btn.addEventListener("click", create_proxy(lambda evt: delete_output(oid)))

def bind_layer_static_events(layer: dict):
    lid = layer["id"]

    sel = get_id(f"act-select-{lid}")
    if sel:
        def h(evt):
            on_layer_act_select_change(lid, evt)
        sel.addEventListener("change", create_proxy(h))

    help_btn = get_id(f"act-help-btn-{lid}")
    if help_btn:
        help_btn.addEventListener("click", create_proxy(open_act_help))

    add_neuron_btn = get_id(f"add-neuron-btn-{lid}")
    if add_neuron_btn:
        add_neuron_btn.addEventListener("click", create_proxy(lambda evt: add_neuron(lid)))

    bind_custom_box_events(lid)

# ── CRUD: inputs ────────────────────────────────────────────────────────────────

def _next_input_name() -> str:
    n = len(inputs) + 1
    return chr((ord('x') - ord('a') + (n - 1)) % 26 + ord('a'))

def add_input(evt=None):
    global input_counter
    input_counter += 1
    iid = input_counter
    inp = {"id": iid, "name": _next_input_name(), "prev_device": None, "prev_channel": None}
    inputs.append(inp)

    container = get_id("inputs-container")
    wrapper = document.createElement("div")
    wrapper.innerHTML = make_input_html(inp)
    container.appendChild(wrapper.firstElementChild)
    bind_input_events(iid)

    if layers:
        for n in layers[0]["neurons"]:
            n["weights"].append(1.0)
        rebuild_layer_eq_html(0)

    update_delete_visibility(inputs, "input")

    for p in training_data:
        p["xs"].setdefault(iid, 0.0)
    render_dataset_header()
    render_dataset_table()
    render_add_point_row()
    refresh_dataset_plot_points()

    def make_plot(iid=iid):
        all_plots[f"plot-in-{iid}"] = plot.plot(f"plot-in-{iid}")
    window.setTimeout(create_proxy(make_plot), 60)

    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def delete_input(iid: int):
    if len(inputs) <= 1:
        return
    idx = next((i for i, inp in enumerate(inputs) if inp["id"] == iid), None)
    if idx is None:
        return
    inp = inputs[idx]

    plot_id = f"plot-in-{iid}"
    _detach_plot_from_device(plot_id, inp.get("prev_device"))
    all_plots.pop(plot_id, None)

    el = get_id(f"item-input-{iid}")
    if el:
        el.remove()

    inputs.pop(idx)
    if layers:
        for n in layers[0]["neurons"]:
            if idx < len(n["weights"]):
                n["weights"].pop(idx)
        rebuild_layer_eq_html(0)

    for p in training_data:
        p["xs"].pop(iid, None)
    render_dataset_header()
    render_dataset_table()
    render_add_point_row()
    refresh_dataset_plot_points()

    update_delete_visibility(inputs, "input")
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

# ── CRUD: neurons ───────────────────────────────────────────────────────────────

def add_neuron(lid: int):
    global neuron_counter
    layer = layer_by_id(lid)
    idx = layer_index_by_id(lid)
    if layer is None or idx is None:
        return

    neuron_counter += 1
    nid = neuron_counter
    n = {"id": nid, "weights": [1.0] * get_source_count(idx), "bias": 0.0}
    layer["neurons"].append(n)

    container = get_id(f"neurons-container-{lid}")
    wrapper = document.createElement("div")
    wrapper.innerHTML = make_neuron_html(n, get_source_labels(idx))
    container.appendChild(wrapper.firstElementChild)
    bind_neuron_events(lid, nid)

    refresh_all_neuron_delete_visibility()
    sync_downstream_after_neuron_count_change(idx)
    update_neuron_usage_state()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def delete_neuron(lid: int, nid: int):
    layer = layer_by_id(lid)
    idx = layer_index_by_id(lid)
    if layer is None or idx is None:
        return
    pos = next((i for i, n in enumerate(layer["neurons"]) if n["id"] == nid), None)
    if pos is None:
        return

    # Deleting the first neuron in a layer removes the whole layer (and its
    # activation block), as long as another layer exists to take its place.
    if pos == 0 and len(layers) > 1:
        delete_layer(lid)
        return

    if len(layer["neurons"]) <= 1:
        return

    el = get_id(f"item-neuron-{nid}")
    if el:
        el.remove()

    layer["neurons"].pop(pos)
    neuron_pre_values.pop(nid, None)
    neuron_post_values.pop(nid, None)

    refresh_all_neuron_delete_visibility()
    sync_downstream_after_neuron_count_change(idx, removed_pos=pos)
    update_neuron_usage_state()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

# ── CRUD: layers ────────────────────────────────────────────────────────────────

def delete_layer(lid: int):
    """Remove an entire layer (all its neurons + its activation block).
    Never removes the last remaining layer. The layer that used to follow
    this one (if any) is reconnected to draw from this layer's predecessor."""
    idx = layer_index_by_id(lid)
    if idx is None or len(layers) <= 1:
        return
    layer = layers[idx]

    for n in layer["neurons"]:
        el = get_id(f"item-neuron-{n['id']}")
        if el:
            el.remove()
        neuron_pre_values.pop(n["id"], None)
        neuron_post_values.pop(n["id"], None)

    col_n = get_id(f"col-neurons-{lid}")
    if col_n:
        col_n.remove()
    col_a = get_id(f"act-col-{lid}")
    if col_a:
        col_a.remove()

    layers.pop(idx)

    # The layer now sitting at `idx` (previously idx+1, if any) needs its
    # weights/labels rebuilt against its new predecessor.
    if idx < len(layers):
        rebuild_layer_eq_html(idx)

    refresh_all_neuron_delete_visibility()
    update_neuron_usage_state()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def add_layer(evt=None):
    global layer_counter
    layer_counter += 1
    lid = layer_counter
    layer = {"id": lid, "neurons": [], "act_fn": "", "custom_activation": {"expr": "x", "pieces": []}}
    layers.append(layer)

    container = get_id("layers-container")
    wrapper = document.createElement("div")
    wrapper.innerHTML = make_layer_neurons_col_html(layer) + make_layer_activation_col_html(layer)
    while wrapper.firstElementChild:
        container.appendChild(wrapper.firstElementChild)

    populate_layer_act_select(lid)
    bind_layer_static_events(layer)

    add_neuron(lid)   # every layer starts with exactly one neuron

    update_neuron_usage_state()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 80)

# ── CRUD: outputs ───────────────────────────────────────────────────────────────

def add_output(evt=None):
    global output_counter
    output_counter += 1
    oid = output_counter
    out = {"id": oid, "prev_device": None, "prev_channel": None}
    outputs.append(out)

    container = get_id("outputs-container")
    wrapper = document.createElement("div")
    wrapper.innerHTML = make_output_html(out)
    container.appendChild(wrapper.firstElementChild)
    bind_output_events(oid)

    def make_plot(oid=oid):
        all_plots[f"plot-out-{oid}"] = plot.plot(f"plot-out-{oid}")
    window.setTimeout(create_proxy(make_plot), 60)

    for p in training_data:
        p["ys"].setdefault(oid, 0.0)

    fit_plot_obj = all_plots.get("plot-fit")
    if fit_plot_obj:
        fit_plot_obj.add_output_trace(oid)

    render_dataset_header()
    render_dataset_table()
    render_add_point_row()
    refresh_dataset_plot_points()

    update_delete_visibility(outputs, "output")
    update_neuron_usage_state()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def delete_output(oid: int):
    if len(outputs) <= 1:
        return
    idx = next((i for i, o in enumerate(outputs) if o["id"] == oid), None)
    if idx is None:
        return
    all_plots.pop(f"plot-out-{oid}", None)

    fit_plot_obj = all_plots.get("plot-fit")
    if fit_plot_obj:
        fit_plot_obj.remove_output_trace(oid)

    el = get_id(f"item-output-{oid}")
    if el:
        el.remove()

    outputs.pop(idx)
    output_values.pop(oid, None)

    for p in training_data:
        p["ys"].pop(oid, None)
    render_dataset_header()
    render_dataset_table()
    render_add_point_row()

    update_delete_visibility(outputs, "output")
    update_neuron_usage_state()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

# ── Device management ────────────────────────────────────────────────────────

async def add_device_chip(dev: Element):
    dl  = get_id("device-list")

    name = dev.name
    chip = document.createElement("div")
    chip.className = "device-row"
    chip.id = f"chip-{name}"
    chip.innerHTML = (
        f'<div class="device-indicator"></div>'
        f'<span class="device-name">{name}</span>'
        f'<button class="btn-disconnect" title="Disconnect">'
        f'    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"'
        f'         stroke="currentColor" stroke-width="2.5">'
        f'        <path d="M18 6L6 18M6 6l12 12"/>'
        f'    </svg>'
        f'</button>'
    )

    disc = chip.querySelector(".btn-disconnect")

    async def make_disc(dev):
        async def handler(evt):
            chip_el = get_id(f"chip-{dev.myble.device.name}")
            if chip_el:
                chip_el.remove()
            await dev.disconnect()
            devices.remove(dev)
            refresh_device_dropdowns()
        return create_proxy(handler)

    disc.addEventListener("click", await make_disc(dev))
    dl.appendChild(chip)
    refresh_device_dropdowns()

async def create_new_device(evt=None):
    new_dev = Element()
    existing_names = [d.name for d in devices]
    await new_dev.connect(existing_names=existing_names)
    print("out of connect")
    if not new_dev.hub or not new_dev.hub.connected:
        return
    devices.append(new_dev)
    await add_device_chip(new_dev)
    refresh_device_dropdowns()

# ── Play / Stop ──────────────────────────────────────────────────────────────

def play_network(evt=None):
    global is_running, run_tick
    is_running = True
    run_tick = 0

    fit_plot_obj = all_plots.get("plot-fit")
    if fit_plot_obj:
        fit_plot_obj.reset_run()

    get_id("play-btn").setAttribute("disabled", "")
    get_id("stop-btn").removeAttribute("disabled")
    document.body.classList.add("running")

    debug_wrap = get_id("debug-toggle-wrap")
    if debug_wrap:
        debug_wrap.classList.remove("hidden")

    asyncio.ensure_future(loop_network())

def stop_network(evt=None):
    global is_running, debug_mode
    is_running = False
    get_id("stop-btn").setAttribute("disabled", "")
    get_id("play-btn").removeAttribute("disabled")
    document.body.classList.remove("running")
    for device in devices:
        try:
            device.stop()
        except Exception as e:
            print("Caught: " + str(e))

    debug_wrap = get_id("debug-toggle-wrap")
    if debug_wrap:
        debug_wrap.classList.add("hidden")

    debug_checkbox = get_id("debug-toggle")
    if debug_checkbox:
        debug_checkbox.checked = False
    debug_mode = False
    redraw_arrows()

async def loop_network():
    while is_running:
        forward()
        if debug_mode:
            redraw_arrows()
        await asyncio.sleep(0.05)

def propagate_layers(source_values: list[float], pre_store: dict, post_store: dict) -> tuple[list[float], list[list[float]]]:
    """Runs source_values through every layer in order, writing each neuron's
    pre/post-activation value into pre_store/post_store (caller passes the
    live global dicts for hardware mode, or fresh local dicts for a
    dataset-training forward pass so the two don't clobber each other).

    Returns (final_layer_outputs, layer_sources) where layer_sources[i] is
    the list of source values that fed layers[i]."""
    layer_sources: list[list[float]] = []
    for layer in layers:
        layer_sources.append(source_values)
        act_fn = layer["act_fn"]
        custom_activation = layer["custom_activation"]
        next_source_values = []
        for n in layer["neurons"]:
            nid = n["id"]
            weighted = sum(
                n["weights"][i] * source_values[i]
                for i in range(min(len(n["weights"]), len(source_values)))
            )
            pre_act = weighted + n["bias"]
            pre_store[nid] = pre_act

            post_act = apply_activation(pre_act, act_fn, custom_activation)
            post_store[nid] = post_act
            next_source_values.append(post_act)
        source_values = next_source_values
    return source_values, layer_sources

def compute_forward() -> list[list[float]]:
    """Read the current input device readings and forward-propagate them
    through every layer, populating input_values / neuron_pre_values /
    neuron_post_values / output_values as a side effect (used by the arrow
    debug display too). Does NOT touch the hardware outputs -- that's
    forward()'s job.

    Returns layer_sources: layer_sources[i] is the list of source values
    that fed layers[i] (raw inputs for layer 0, the previous layer's
    post-activations otherwise) -- backpropagate() needs this to compute
    weight gradients."""
    # 1. Collect input values
    for inp in inputs:
        iid = inp["id"]
        dev_el  = get_id(f"dev-in-{iid}")
        chan_el = get_id(f"chan-in-{iid}")
        dev_name = dev_el.value if dev_el else ""
        channel  = chan_el.value if chan_el else ""
        dev = device_by_name(dev_name)
        try:
            val = float(dev.state[channel]) if dev and channel else 0.0
        except (KeyError, TypeError, ValueError):
            val = 0.0
        input_values[iid] = val

    # 2. Forward-propagate through every layer in order
    source_values = [input_values.get(inp["id"], 0.0) for inp in inputs]
    _, layer_sources = propagate_layers(source_values, neuron_pre_values, neuron_post_values)

    # 3. Route each output from the last layer's neuron at the same position
    last_layer_neurons = layers[-1]["neurons"] if layers else []
    for idx, out in enumerate(outputs):
        oid = out["id"]
        val = neuron_post_values.get(last_layer_neurons[idx]["id"], 0.0) if idx < len(last_layer_neurons) else 0.0
        output_values[oid] = val

    return layer_sources

def forward():
    global run_tick
    compute_forward()
    run_tick += 1

    for idx, out in enumerate(outputs):
        oid = out["id"]
        y_val = output_values.get(oid, 0.0)
        result = int(y_val)
        dev_el  = get_id(f"dev-out-{oid}")
        chan_el = get_id(f"chan-out-{oid}")
        dev_name = dev_el.value if dev_el else ""
        channel  = chan_el.value if chan_el else ""
        run_output(channel, dev_name, result)

        reading_el = get_id(f"reading-out-{oid}")
        if reading_el:
            reading_el.textContent = f"{y_val:.2f}"

        plot_obj = all_plots.get(f"plot-out-{oid}")
        if plot_obj:
            update = plot_obj.addPoints(1, [y_val])
            plot_obj.updatePlot(update)

    fit_plot_obj = all_plots.get("plot-fit")
    if fit_plot_obj and inputs and outputs:
        x_val = input_values.get(inputs[0]["id"], 0.0)
        for out in outputs:
            fit_plot_obj.add_run_point(out["id"], x_val, output_values.get(out["id"], 0.0))

def randomize_weights(evt=None):
    """Give every neuron in every layer a fresh random weight vector + bias."""
    global loss_history
    for layer in layers:
        for n in layer["neurons"]:
            n["weights"] = [random.uniform(-1.0, 1.0) for _ in n["weights"]]
            n["bias"] = random.uniform(-1.0, 1.0)
    for idx in range(len(layers)):
        rebuild_layer_eq_html(idx)
    redraw_arrows()

    loss_history = []
    loss_plot_obj = all_plots.get("plot-loss")
    if loss_plot_obj:
        loss_plot_obj.reset()

# ââ Dataset-based training (normalized internally; live mode untouched) âââââ

def _col_stats(vals: list[float]) -> tuple[float, float]:
    """(mean, std) over a single dataset column, std falling back to 1.0
    whenever there's fewer than 2 points or zero spread."""
    if len(vals) < 2:
        return 0.0, 1.0
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) or 1.0
    return mean, std

def _dataset_stats():
    """(x_stats, y_stats): x_stats maps each CURRENT input id -> (mean, std)
    over that column's values across training_data (missing values in a
    point default to 0.0); y_stats does the same per output id."""
    x_stats = {inp["id"]: _col_stats([p["xs"].get(inp["id"], 0.0) for p in training_data])
               for inp in inputs}
    y_stats = {out["id"]: _col_stats([p["ys"].get(out["id"], 0.0) for p in training_data])
               for out in outputs}
    return x_stats, y_stats

def normalize_x(x, iid, x_stats):
    x_mean, x_std = x_stats.get(iid, (0.0, 1.0))
    return (x - x_mean) / x_std if normalize_enabled else x

def normalize_y(y, oid, y_stats):
    y_mean, y_std = y_stats.get(oid, (0.0, 1.0))
    return (y - y_mean) / y_std if normalize_enabled else y

MAX_GRAD_NORM = 5.0

def _clip_grad(value: float) -> float:
    if value > MAX_GRAD_NORM:
        return MAX_GRAD_NORM
    if value < -MAX_GRAD_NORM:
        return -MAX_GRAD_NORM
    return value

def _unnormalize_weights(x_stats, y_stats):
    """Fold the dataset's per-input/per-output normalization into the
    network's stored weights/biases, converting them from the normalized
    space used DURING training (see train()) into the raw space that
    live/hardware forward() reads directly and unmodified. Without this, a
    network trained on normalized data would predict wildly wrong raw
    values in Play mode -- the whole point of training in normalized
    space is gradient stability, not a permanent change of the model's
    units.

    Folding each input's x-scale into the first layer's matching weight is
    exact for any activation function (it only rescales the pre-activation
    input). Folding an output's y-scale into the matching last-layer
    neuron's weights is only exact when the last layer's activation is
    identity ("None") -- for a nonlinear final activation there is no
    weight-only equivalent, so that step is skipped for that neuron and
    live-mode output stays in normalized-ish units in that edge case.
    A last-layer neuron beyond the current number of outputs has no
    matching output stats and is left untouched (it isn't wired to
    anything yet, per update_neuron_usage_state())."""
    if not layers:
        return
    input_ids = [inp["id"] for inp in inputs]

    first_layer = layers[0]
    for n in first_layer["neurons"]:
        old_weights = list(n["weights"])
        bias_adjust = 0.0
        new_weights = []
        for i, w in enumerate(old_weights):
            x_mean, x_std = x_stats.get(input_ids[i], (0.0, 1.0)) if i < len(input_ids) else (0.0, 1.0)
            new_weights.append(w / x_std)
            bias_adjust += w * x_mean / x_std
        n["bias"] = n["bias"] - bias_adjust
        n["weights"] = new_weights

    last_layer = layers[-1]
    if last_layer["act_fn"] in ("", None):
        output_ids = [out["id"] for out in outputs]
        for idx, n in enumerate(last_layer["neurons"]):
            if idx >= len(output_ids):
                continue
            y_mean, y_std = y_stats.get(output_ids[idx], (0.0, 1.0))
            n["bias"] = y_mean + y_std * n["bias"]
            n["weights"] = [w * y_std for w in n["weights"]]

def _renormalize_weights(x_stats, y_stats):
    """Inverse of _unnormalize_weights(): converts the canonical raw-space
    weights/biases into the normalized-space equivalents train_epoch()
    expects, immediately before a training run. Must undo the two folds
    in reverse order (last layer's y-fold, then the first layer's
    x-fold) to exactly invert _unnormalize_weights()."""
    if not layers:
        return

    last_layer = layers[-1]
    if last_layer["act_fn"] in ("", None):
        output_ids = [out["id"] for out in outputs]
        for idx, n in enumerate(last_layer["neurons"]):
            if idx >= len(output_ids):
                continue
            y_mean, y_std = y_stats.get(output_ids[idx], (0.0, 1.0))
            old_weights = list(n["weights"])
            n["bias"] = (n["bias"] - y_mean) / y_std
            n["weights"] = [w / y_std for w in old_weights]

    input_ids = [inp["id"] for inp in inputs]
    first_layer = layers[0]
    for n in first_layer["neurons"]:
        old_weights = list(n["weights"])
        bias_adjust = 0.0
        new_weights = []
        for i, w in enumerate(old_weights):
            x_mean, x_std = x_stats.get(input_ids[i], (0.0, 1.0)) if i < len(input_ids) else (0.0, 1.0)
            new_weights.append(w * x_std)
            bias_adjust += w * x_mean
        n["bias"] = n["bias"] + bias_adjust
        n["weights"] = new_weights

def train_epoch(lr: float, x_stats, y_stats) -> float:
    """One pass over the whole dataset: accumulate gradients per point (and
    per output, positionally paired with the last layer's neurons -- same
    convention as compute_forward()), then apply one averaged update per
    weight/bias. Returns the mean per-point-per-output loss. Assumes the
    network's weights are already in normalized space (see train())."""
    if not training_data or not layers or not outputs:
        return 0.0

    input_ids = [inp["id"] for inp in inputs]
    output_ids = [out["id"] for out in outputs]
    act_fns = [layer["act_fn"] for layer in layers]
    customs = [layer["custom_activation"] for layer in layers]

    weight_grads = [[[0.0] * len(n["weights"]) for n in layer["neurons"]] for layer in layers]
    bias_grads = [[0.0 for _ in layer["neurons"]] for layer in layers]

    last_layer_neurons = layers[-1]["neurons"]
    total_loss = 0.0
    loss_terms = 0

    for point in training_data:
        xs_n = [normalize_x(point["xs"].get(iid, 0.0), iid, x_stats) for iid in input_ids]

        pre_store, post_store = {}, {}
        final_values, layer_sources = propagate_layers(xs_n, pre_store, post_store)

        deltas = [0.0] * len(last_layer_neurons)
        for oidx, oid in enumerate(output_ids):
            if oidx >= len(last_layer_neurons):
                break
            n = last_layer_neurons[oidx]
            pred_n = final_values[oidx] if oidx < len(final_values) else 0.0
            y_n = normalize_y(point["ys"].get(oid, 0.0), oid, y_stats)

            err = pred_n - y_n
            total_loss += err ** 2
            loss_terms += 1

            deriv = apply_activation_derivative(
                pre_store.get(n["id"], 0.0), pred_n, act_fns[-1], customs[-1])
            deltas[oidx] = 2.0 * err * deriv

        for layer_idx in reversed(range(len(layers))):
            layer = layers[layer_idx]
            neurons = layer["neurons"]
            sources = layer_sources[layer_idx] if layer_idx < len(layer_sources) else []
            next_deltas = [0.0] * len(sources)

            for i, n in enumerate(neurons):
                d = deltas[i] if i < len(deltas) else 0.0
                if d == 0.0:
                    continue
                for j in range(min(len(n["weights"]), len(sources))):
                    next_deltas[j] += d * n["weights"][j]
                    weight_grads[layer_idx][i][j] += d * sources[j]
                bias_grads[layer_idx][i] += d

            if layer_idx > 0:
                prev_layer = layers[layer_idx - 1]
                new_deltas = []
                for i, pn in enumerate(prev_layer["neurons"]):
                    pre = pre_store.get(pn["id"], 0.0)
                    post = post_store.get(pn["id"], 0.0)
                    deriv = apply_activation_derivative(pre, post, act_fns[layer_idx - 1], customs[layer_idx - 1])
                    raw = next_deltas[i] if i < len(next_deltas) else 0.0
                    new_deltas.append(raw * deriv)
                deltas = new_deltas

    n_points = len(training_data)
    for layer_idx, layer in enumerate(layers):
        for i, n in enumerate(layer["neurons"]):
            for j in range(len(n["weights"])):
                grad = _clip_grad(weight_grads[layer_idx][i][j] / n_points)
                n["weights"][j] -= lr * grad
            bgrad = _clip_grad(bias_grads[layer_idx][i] / n_points)
            n["bias"] -= lr * bgrad

    return total_loss / loss_terms if loss_terms else 0.0

def _append_loss(loss: float):
    loss_history.append(loss)
    loss_plot_obj = all_plots.get("plot-loss")
    if loss_plot_obj:
        update = loss_plot_obj.add_point(loss)
        loss_plot_obj.updatePlot(update)

def train(epochs: int, lr: float):
    if not training_data:
        print("Add at least one data point before training.")
        return

    x_stats, y_stats = _dataset_stats()
    if normalize_enabled:
        _renormalize_weights(x_stats, y_stats)

    for _ in range(epochs):
        loss = train_epoch(lr, x_stats, y_stats)
        _append_loss(loss)

    if normalize_enabled:
        _unnormalize_weights(x_stats, y_stats)

    for idx in range(len(layers)):
        rebuild_layer_eq_html(idx)
    print(f"Trained {epochs} epoch(s), lr={lr}. Loss: {loss_history[-1] if loss_history else 0.0:.4f}")

def train_step(evt=None):
    train(epochs=1, lr=get_learning_rate())

def train_30_epochs(evt=None):
    train(epochs=30, lr=get_learning_rate())

# ââ Dataset entry (Phase 1) âââââââââââââââââââââââââââââââââââââââââââââââââââ

def render_dataset_header():
    container = get_id("dataset-header")
    if not container:
        return
    html = ""
    for inp in inputs:
        html += f'<span class="dataset-row-header-x">{inp["name"]}</span>'
    for idx, out in enumerate(outputs):
        html += f'<span class="dataset-row-header-y">y{idx + 1}</span>'
    html += "<span></span>"
    container.innerHTML = html

def render_dataset_table():
    container = get_id("dataset-rows")
    if not container:
        return
    html = ""
    for p in training_data:
        pid = p["id"]
        html += f'<div class="dataset-row" data-id="{pid}">'
        for inp in inputs:
            iid = inp["id"]
            val = p["xs"].get(iid, 0.0)
            html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-x" '
                      f'id="point-x-{pid}-{iid}" value="{val}" />')
        for out in outputs:
            oid = out["id"]
            val = p["ys"].get(oid, 0.0)
            html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-y" '
                      f'id="point-y-{pid}-{oid}" value="{val}" />')
        html += f'<button class="btn-remove-point" id="del-point-{pid}" title="Remove point">{delete_x_svg()}</button>'
        html += "</div>"
    container.innerHTML = html
    for p in training_data:
        bind_dataset_row_events(p["id"])

def bind_dataset_row_events(pid: int):
    for inp in inputs:
        iid = inp["id"]
        x_el = get_id(f"point-x-{pid}-{iid}")
        if x_el:
            def hx(evt, pid=pid, iid=iid):
                p = next((pt for pt in training_data if pt["id"] == pid), None)
                try:
                    if p:
                        p["xs"][iid] = float(evt.target.value)
                        refresh_dataset_plot_points()
                except (ValueError, TypeError):
                    pass
            x_el.addEventListener("input", create_proxy(hx))

    for out in outputs:
        oid = out["id"]
        y_el = get_id(f"point-y-{pid}-{oid}")
        if y_el:
            def hy(evt, pid=pid, oid=oid):
                p = next((pt for pt in training_data if pt["id"] == pid), None)
                try:
                    if p:
                        p["ys"][oid] = float(evt.target.value)
                        refresh_dataset_plot_points()
                except (ValueError, TypeError):
                    pass
            y_el.addEventListener("input", create_proxy(hy))

    del_btn = get_id(f"del-point-{pid}")
    if del_btn:
        del_btn.addEventListener("click", create_proxy(lambda evt, pid=pid: remove_data_point(pid)))

def refresh_dataset_plot_points():
    """Push each output's Data trace: y = that output's column, x = the
    FIRST input's column (the fit graph's x-axis is fixed to it)."""
    fit_plot_obj = all_plots.get("plot-fit")
    if not fit_plot_obj or not inputs:
        return
    first_iid = inputs[0]["id"]
    xs = [p["xs"].get(first_iid, 0.0) for p in training_data]
    for out in outputs:
        oid = out["id"]
        ys = [p["ys"].get(oid, 0.0) for p in training_data]
        fit_plot_obj.update_data_points(oid, xs, ys)

def render_add_point_row():
    container = get_id("dataset-add-row")
    if not container:
        return
    html = ""
    for inp in inputs:
        iid = inp["id"]
        html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-x" '
                  f'id="new-point-x-{iid}" placeholder="{inp["name"]}" />')
    for idx, out in enumerate(outputs):
        oid = out["id"]
        html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-y" '
                  f'id="new-point-y-{oid}" placeholder="y{idx + 1}" />')
    html += ('<button class="btn-add-point" id="add-point-btn" title="Add this point to the dataset">'
             '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="2.5"><path d="M12 5v14M5 12h14"/></svg>'
             'Add point</button>')
    container.innerHTML = html
    btn = get_id("add-point-btn")
    if btn:
        btn.addEventListener("click", create_proxy(on_add_point_click))

def add_data_point(xs: dict, ys: dict):
    global data_point_counter
    data_point_counter += 1
    training_data.append({"id": data_point_counter, "xs": dict(xs), "ys": dict(ys)})
    render_dataset_table()
    refresh_dataset_plot_points()

def remove_data_point(pid: int):
    idx = next((i for i, p in enumerate(training_data) if p["id"] == pid), None)
    if idx is None:
        return
    training_data.pop(idx)
    render_dataset_table()
    refresh_dataset_plot_points()

def on_add_point_click(evt=None):
    xs = {}
    for inp in inputs:
        iid = inp["id"]
        el = get_id(f"new-point-x-{iid}")
        try:
            xs[iid] = float(el.value) if el and el.value.strip() != "" else None
        except ValueError:
            xs[iid] = None

    ys = {}
    for out in outputs:
        oid = out["id"]
        el = get_id(f"new-point-y-{oid}")
        try:
            ys[oid] = float(el.value) if el and el.value.strip() != "" else None
        except ValueError:
            ys[oid] = None

    if any(v is None for v in xs.values()) or any(v is None for v in ys.values()):
        print("Enter every x and y value before adding a point.")
        return

    add_data_point(xs, ys)
    for inp in inputs:
        el = get_id(f"new-point-x-{inp['id']}")
        if el:
            el.value = ""
    for out in outputs:
        el = get_id(f"new-point-y-{out['id']}")
        if el:
            el.value = ""

def run_output(variable, dev_name, value):
    device = device_by_name(dev_name)
    if not device:
        return
    if "Speed" in variable:
        if value > 100:
            value = 100
        elif value < -100:
            value = -100
    if variable == "Speed":
        device.set_speed(value)
    elif variable == "LeftSpeed":
        device.set_speedL(value)
    elif variable == "RightSpeed":
        device.set_speedR(value)
    elif variable == "BothSpeed":
        device.set_speed(value)
    elif variable == "LightColor":
        device.set_light(variable, value)
    elif variable == "LightPattern":
        device.set_light(variable, value)
    elif variable == "LightIntensity":
        if value > 100:
            value = 100
        elif value < 0:
            value = 0
        device.set_light(variable, value)
    elif variable == "BeepPattern":
        device.set_beep(variable, value)
    elif variable == "BeepFrequency":
        if value < 0:
            value = 0
        elif value > 2700:
            value = 2700
        device.set_beep(variable, value)
    else:
        print("Cannot set " + str(variable))

# ── Activation help popover (shared by every layer's "?" button) ────────────

def open_act_help(evt=None):
    get_id("act-help-popover").classList.remove("hidden")

def close_act_help(evt=None):
    get_id("act-help-popover").classList.add("hidden")

# ── Custom activation piecewise editor (per layer) ───────────────────────────

def make_piece_html(p: dict) -> str:
    pid = p["id"]
    def op_opts(current):
        out = ""
        for val, sym in (("<", "&lt;"), ("<=", "&le;"), (">", "&gt;"), (">=", "&ge;")):
            sel = "selected" if val == current else ""
            out += f'<option value="{val}" {sel}>{sym}</option>'
        return out
    return f"""
<div class="custom-piece-row" data-piece="{pid}">
    <input type="text" class="custom-eq-input small" id="piece-expr-{pid}" value="{p['expr']}" />
    <span class="piece-if">if</span>
    <input type="text" class="piece-num-input" id="piece-lo-{pid}" placeholder="-∞" value="{p['lo']}" />
    <select class="piece-op-select" id="piece-lo-op-{pid}">{op_opts(p['lo_op'])}</select>
    <span class="piece-x">x</span>
    <select class="piece-op-select" id="piece-hi-op-{pid}">{op_opts(p['hi_op'])}</select>
    <input type="text" class="piece-num-input" id="piece-hi-{pid}" placeholder="∞" value="{p['hi']}" />
    <button class="btn-remove-piece" id="remove-piece-{pid}">×</button>
</div>
"""

def render_custom_pieces(lid: int):
    layer = layer_by_id(lid)
    if not layer:
        return
    list_el    = get_id(f"custom-pieces-list-{lid}")
    wrap_el    = get_id(f"custom-pieces-wrap-{lid}")
    default_el = get_id(f"custom-default-expr-{lid}")
    if not list_el:
        return
    pieces = layer["custom_activation"]["pieces"]
    if pieces:
        wrap_el.classList.remove("hidden")
        default_el.classList.add("hidden")
        list_el.innerHTML = "".join(make_piece_html(p) for p in pieces)
        for p in pieces:
            bind_piece_events(lid, p["id"])
    else:
        wrap_el.classList.add("hidden")
        default_el.classList.remove("hidden")
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def find_piece(lid: int, pid: int):
    layer = layer_by_id(lid)
    if not layer:
        return None
    return next((p for p in layer["custom_activation"]["pieces"] if p["id"] == pid), None)

def bind_piece_events(lid: int, pid: int):
    def bind(elid, evname, key):
        el = get_id(elid)
        if not el:
            return
        def h(evt):
            p = find_piece(lid, pid)
            if p:
                p[key] = evt.target.value
        el.addEventListener(evname, create_proxy(h))

    bind(f"piece-expr-{pid}",  "input",  "expr")
    bind(f"piece-lo-{pid}",    "input",  "lo")
    bind(f"piece-hi-{pid}",    "input",  "hi")
    bind(f"piece-lo-op-{pid}", "change", "lo_op")
    bind(f"piece-hi-op-{pid}", "change", "hi_op")

    rm = get_id(f"remove-piece-{pid}")
    if rm:
        def h(evt):
            remove_piece(lid, pid)
        rm.addEventListener("click", create_proxy(h))

def add_piece(lid: int):
    global piece_counter
    layer = layer_by_id(lid)
    if not layer:
        return
    piece_counter += 1
    pid = piece_counter
    seed_expr = "x"
    if not layer["custom_activation"]["pieces"]:
        default_el = get_id(f"custom-default-expr-{lid}")
        seed_expr = default_el.value if default_el else layer["custom_activation"]["expr"]
    layer["custom_activation"]["pieces"].append({
        "id": pid, "expr": seed_expr,
        "lo": "", "lo_op": "<", "hi": "", "hi_op": "<",
    })
    render_custom_pieces(lid)

def remove_piece(lid: int, pid: int):
    layer = layer_by_id(lid)
    if not layer:
        return
    layer["custom_activation"]["pieces"] = [
        p for p in layer["custom_activation"]["pieces"] if p["id"] != pid
    ]
    render_custom_pieces(lid)

def bind_custom_box_events(lid: int):
    default_el = get_id(f"custom-default-expr-{lid}")
    if default_el:
        def h(evt):
            layer = layer_by_id(lid)
            if layer:
                layer["custom_activation"]["expr"] = evt.target.value
        default_el.addEventListener("input", create_proxy(h))

    add_btn = get_id(f"add-piece-btn-{lid}")
    if add_btn:
        add_btn.addEventListener("click", create_proxy(lambda evt: add_piece(lid)))

def on_layer_act_select_change(lid: int, evt):
    layer = layer_by_id(lid)
    if not layer:
        return
    val = evt.target.value
    layer["act_fn"] = val

    box = get_id(f"custom-act-box-{lid}")
    if box:
        is_custom = (val == "custom")
        box.classList.toggle("hidden", not is_custom)
        get_id(f"act-box-{lid}").classList.toggle("has-custom", is_custom)
        get_id(f"act-col-{lid}").classList.toggle("act-col-custom", is_custom)
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

# ── Zoom (network diagram spacing/box sizes only) ─────────────────────────────

def apply_zoom():
    wrap = get_id("network-wrap")
    if wrap:
        wrap.style.setProperty("--zoom", str(zoom_level))
        # Text stays full-size down to the threshold, then shrinks in step
        # with everything else below it.
        if zoom_level >= TEXT_ZOOM_THRESHOLD:
            text_zoom = 1.0
        else:
            text_zoom = zoom_level / TEXT_ZOOM_THRESHOLD
        wrap.style.setProperty("--text-zoom", str(round(text_zoom, 3)))

    label = get_id("zoom-level-label")
    if label:
        label.textContent = f"{round(zoom_level * 100)}%"

    out_btn = get_id("zoom-out-btn")
    if out_btn:
        if zoom_level <= ZOOM_MIN + 1e-9:
            out_btn.setAttribute("disabled", "")
        else:
            out_btn.removeAttribute("disabled")

    in_btn = get_id("zoom-in-btn")
    if in_btn:
        if zoom_level >= ZOOM_MAX - 1e-9:
            in_btn.setAttribute("disabled", "")
        else:
            in_btn.removeAttribute("disabled")

    # Layout changed -- re-measure anchor points once the browser has
    # applied the new spacing/sizes.
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def zoom_out(evt=None):
    global zoom_level
    zoom_level = round(max(ZOOM_MIN, zoom_level - ZOOM_STEP), 2)
    apply_zoom()

def zoom_in(evt=None):
    global zoom_level
    zoom_level = round(min(ZOOM_MAX, zoom_level + ZOOM_STEP), 2)
    apply_zoom()

# ── Resize observer ────────────────────────────────────────────────────────────

def setup_resize_observer():
    wrap = get_id("network-wrap")
    if not wrap:
        return
    def on_resize(entries, observer):
        redraw_arrows()
    observer = window.ResizeObserver.new(create_proxy(on_resize))
    observer.observe(wrap)

# ── Static wiring ──────────────────────────────────────────────────────────────

@when("click", "#add-device-btn")
async def _on_add_device(evt):
    await create_new_device()

@when("click", "#add-input-btn")
def _on_add_input(evt):
    add_input()

@when("click", "#add-layer-btn")
def _on_add_layer(evt):
    add_layer()

@when("click", "#add-output-btn")
def _on_add_output(evt):
    add_output()

@when("click", "#randomize-btn")
def _on_randomize(evt):
    randomize_weights()

@when("click", "#step-btn")
def _on_step(evt):
    train_step()

@when("click", "#train30-btn")
def _on_train30(evt):
    train_30_epochs()

@when("click", "#clear-fit-btn")
def _on_clear_fit(evt):
    fit_plot_obj = all_plots.get("plot-fit")
    if fit_plot_obj:
        fit_plot_obj.reset_run()

@when("click", "#play-btn")
def _on_play(evt):
    play_network()

@when("click", "#stop-btn")
def _on_stop(evt):
    stop_network()

@when("click", "#close-act-help-btn")
def _on_close_act_help(evt):
    close_act_help()

@when("click", "#zoom-out-btn")
def _on_zoom_out(evt):
    zoom_out()

@when("click", "#zoom-in-btn")
def _on_zoom_in(evt):
    zoom_in()

@when("change", "#debug-toggle")
def _on_debug_toggle(evt):
    global debug_mode
    debug_mode = evt.target.checked
    redraw_arrows()

# ── Boot ───────────────────────────────────────────────────────────────────────

def boot():
    get_id("loading-splash").style.display = "none"
    get_id("page-wrap").style.display = "flex"

    all_plots["plot-fit"] = plot.fit_plot("plot-fit")
    all_plots["plot-loss"] = plot.loss_plot("plot-loss")

    # one of each on load: 1 input, 1 layer (1 neuron + activation), 1 output
    add_input()
    add_layer()
    add_output()

    apply_zoom()
    setup_resize_observer()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 150)

boot()