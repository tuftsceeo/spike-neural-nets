import asyncio
import json
import math
import random

from pyodide.ffi import create_proxy
from js import document, Plotly, JSON


def to_js(obj):
    """Convert a Python dict/list structure into a real JS object/array
    (Plotly.js chokes on PyProxy dicts - it needs actual JS objects)."""
    return JSON.parse(json.dumps(obj))


# ─────────────────────────────────────────────────────────────────
# DOM shortcuts
# ─────────────────────────────────────────────────────────────────

def el(id_):
    return document.getElementById(id_)

p1x_input = el("p1x")
p1y_input = el("p1y")
p2x_input = el("p2x")
p2y_input = el("p2y")
error_select = el("error-select")
custom_error_input = el("custom-error-input")
lr_input = el("lr-input")
randomize_btn = el("randomize-btn")
reset_btn = el("reset-btn")
setup_row_el = el("setup-row")
back_epoch_btn = el("back-epoch-btn")
back_step_btn = el("back-step-btn")
step_btn = el("step-btn")
epoch_btn = el("epoch-btn")
play_pause_btn = el("play-pause-btn")
toggle_explanation_btn = el("toggle-explanation-btn")
flowchart_panel_el = el("flowchart-panel")
prediction_plot_el = el("prediction-plot")
loss_plot_el = el("loss-plot")
live_network_eq_el = el("live-network-eq")

w_slice_panel_el = el("w-slice-panel")
b_slice_panel_el = el("b-slice-panel")
w_slice_plot_el = el("w-slice-plot")
b_slice_plot_el = el("b-slice-plot")

INIT_RANGE = (-1, 1)

# ─────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────

class TrainingState:
    def __init__(self):
        self.p1 = (0.0, 0.0)
        self.p2 = (0.0, 0.0)
        self.error_key = "mse"
        self.custom_error_expr = ""
        self.lr = 0.1

        self.w = 0.0
        self.b = 0.0
        self.epoch = 0

        self.forward_result = None
        self.grad_result = None
        self.update_result = None

        self.loss_history = []
        self.step_index = 0
        self.initialized = False
        self.playing = False

        self.last_w_dir = None
        self.last_b_dir = None

        # Snapshot stack for backward step/epoch. Each entry fully
        # describes the state right after some do_step() call finished.
        self.history = []


state = TrainingState()

# ─────────────────────────────────────────────────────────────────
# Pure math
# ─────────────────────────────────────────────────────────────────

def _mae_grad(pred, y):
    if pred == y:
        return 0.0
    return 1.0 if pred > y else -1.0


ERROR_FUNCTIONS = {
    "mse": (lambda pred, y: (pred - y) ** 2, lambda pred, y: 2 * (pred - y)),
    "mae": (lambda pred, y: abs(pred - y), _mae_grad),
}

SAFE_NAMES = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": lambda x: math.sqrt(x) if x >= 0 else float("nan"),
    "exp": math.exp,
    "log": lambda x: math.log(x) if x > 0 else float("nan"),
    "sin": math.sin,
    "cos": math.cos,
}


def safe_eval(expr, pred, y):
    ns = dict(SAFE_NAMES)
    ns["pred"] = pred
    ns["y"] = y
    return eval(expr, {"__builtins__": {}}, ns)


def custom_error_fn(expr):
    def e(pred, y):
        return safe_eval(expr, pred, y)

    def grad(pred, y):
        h = 1e-4
        return (e(pred + h, y) - e(pred - h, y)) / (2 * h)

    return e, grad


def resolve_error_fns():
    if state.error_key == "custom":
        return custom_error_fn(state.custom_error_expr)
    return ERROR_FUNCTIONS[state.error_key]


def error_label():
    return {"mse": "MSE", "mae": "MAE", "custom": "Custom"}[state.error_key]


def forward(x1, y1, x2, y2, w, b):
    return {"pred1": w * x1 + b, "pred2": w * x2 + b}


def compute_error_and_grads(x1, y1, pred1, x2, y2, pred2, e_fn, grad_fn):
    e1, e2 = e_fn(pred1, y1), e_fn(pred2, y2)
    E = (e1 + e2) / 2
    g1, g2 = grad_fn(pred1, y1), grad_fn(pred2, y2)
    dE_dw = (g1 * x1 + g2 * x2) / 2
    dE_db = (g1 + g2) / 2
    return {"E": E, "e1": e1, "e2": e2, "dE_dw": dE_dw, "dE_db": dE_db}


def direction_arrow(delta):
    if delta > 0:
        return "down"
    if delta < 0:
        return "up"
    return "flat"


def compute_update(w, b, dE_dw, dE_db, lr):
    delta_w, delta_b = lr * dE_dw, lr * dE_db
    return {
        "delta_w": delta_w,
        "delta_b": delta_b,
        "w_new": w - delta_w,
        "b_new": b - delta_b,
        "w_dir": direction_arrow(delta_w),
        "b_dir": direction_arrow(delta_b),
    }


ARROW_GLYPH = {"up": "▲", "down": "▼", "flat": "–"}


def fmt(x, n=3):
    try:
        return f"{x:.{n}f}"
    except (ValueError, TypeError):
        return str(x)


# ─────────────────────────────────────────────────────────────────
# Flowchart skeleton — built ONCE. Every step after that only updates
# text content and swaps state-inactive/state-active/state-done
# classes; boxes never get destroyed or recreated. A single SVG
# overlay spans the whole block so arrows can run from the per-point
# boxes all the way down through the flowchart rows.
# ─────────────────────────────────────────────────────────────────

def make_el(tag, class_name=None, text=None):
    e = document.createElement(tag)
    if class_name:
        e.className = class_name
    if text is not None:
        e.textContent = text
    return e


def set_tip(elem, text):
    elem.classList.add("tip")
    elem.setAttribute("data-tip", text)
    return elem


def build_flow_skeleton():
    flowchart_panel_el.innerHTML = ""

    block = make_el("div", "epoch-block")
    block.id = "epoch-block"

    # ── per-point calculation boxes (kept visually separate) ──────
    point_row = make_el("div", "point-row")
    for label in ("p1", "p2"):
        box = make_el("div", "point-calc-box state-inactive")
        box.id = f"point-box-{label}"

        diagram = make_el("div", "net-diagram")
        x_node = make_el("div", "node-card input-node", text="x = –")
        x_node.id = f"point-x-{label}"
        set_tip(x_node, "x \u2014 this point's input value")
        diagram.appendChild(x_node)
        diagram.appendChild(make_el("div", "flow-arrow", text="→"))
        lin = make_el("div", "eq-node linear-node", text="w\u00b7x + b")
        lin.id = f"point-lin-{label}"
        set_tip(lin, "\u0177 = w\u00b7x + b \u2014 forward pass")
        diagram.appendChild(lin)
        diagram.appendChild(make_el("div", "eq-op", text="\u2192"))
        out = make_el("div", "node-card output-node")
        out.id = f"point-out-{label}"
        out.innerHTML = "pred = – <span class='target-note'>(target –)</span>"
        set_tip(out, "\u0177 \u2014 the model's prediction here")
        diagram.appendChild(out)

        box.appendChild(diagram)
        point_row.appendChild(box)
    block.appendChild(point_row)

    flow_wrap = make_el("div", "flowchart-wrap")
    flow_wrap.id = "flowchart-wrap"

    error_row = make_el("div", "flow-row flow-row-error state-inactive")
    error_row.id = "flow-row-error"
    error_node = make_el("div", "eq-node error-node", text="E = –")
    error_node.id = "error-node"
    set_tip(error_node, "E \u2014 average error across both points")
    error_row.appendChild(error_node)
    flow_wrap.appendChild(error_row)

    grad_row = make_el("div", "flow-row flow-row-grads state-inactive")
    grad_row.id = "flow-row-grads"
    dw = make_el("div", "eq-node grad-node w-color", text="dE/dw = –")
    dw.id = "grad-w-node"
    set_tip(dw, "w's gradient \u2014 how error responds to w")
    db = make_el("div", "eq-node grad-node b-color", text="dE/db = –")
    db.id = "grad-b-node"
    set_tip(db, "b's gradient \u2014 how error responds to b")
    grad_row.appendChild(dw)
    grad_row.appendChild(db)
    flow_wrap.appendChild(grad_row)

    update_row = make_el("div", "flow-row flow-row-updates state-inactive")
    update_row.id = "flow-row-updates"

    cell_w = make_el("div", "update-cell")
    node_w = make_el("div", "eq-node update-node w-color", text="\u0394w = –")
    node_w.id = "update-w-node"
    set_tip(node_w, "\u0394w = learning rate \u00d7 w's gradient")
    cell_w.appendChild(node_w)

    cell_b = make_el("div", "update-cell")
    node_b = make_el("div", "eq-node update-node b-color", text="\u0394b = –")
    node_b.id = "update-b-node"
    set_tip(node_b, "\u0394b = learning rate \u00d7 b's gradient")
    cell_b.appendChild(node_b)

    update_row.appendChild(cell_w)
    update_row.appendChild(cell_b)
    flow_wrap.appendChild(update_row)

    final_row = make_el("div", "flow-row flow-row-final state-inactive")
    final_row.id = "flow-row-final"
    final_w = make_el("div", "final-param w-color")
    final_w.id = "final-w-node"
    final_w.innerHTML = "<span class='dir-arrow'>–</span> w = –"
    set_tip(final_w, "new w = old w \u2212 \u0394w")
    final_b = make_el("div", "final-param b-color")
    final_b.id = "final-b-node"
    final_b.innerHTML = "<span class='dir-arrow'>–</span> b = –"
    set_tip(final_b, "new b = old b \u2212 \u0394b")
    final_row.appendChild(final_w)
    final_row.appendChild(final_b)
    flow_wrap.appendChild(final_row)

    block.appendChild(flow_wrap)
    flowchart_panel_el.appendChild(block)


def set_state(elem_id, state_name):
    e = el(elem_id)
    if e is None:
        return
    e.classList.remove("state-inactive", "state-active", "state-done")
    e.classList.add(f"state-{state_name}")


def set_all_states(point_state, error_state, grad_state, update_state, final_state):
    set_state("point-box-p1", point_state)
    set_state("point-box-p2", point_state)
    set_state("flow-row-error", error_state)
    set_state("flow-row-grads", grad_state)
    set_state("flow-row-updates", update_state)
    set_state("flow-row-final", final_state)


def render_point_diagrams(fwd, w, b):
    x1, y1 = state.p1
    x2, y2 = state.p2
    specs = [("p1", x1, y1, fwd["pred1"]), ("p2", x2, y2, fwd["pred2"])]
    for label, x, y, pred in specs:
        el(f"point-x-{label}").textContent = f"x = {fmt(x)}"
        el(f"point-lin-{label}").innerHTML = (
            f"<span class='w-color'>{fmt(w)}</span>\u00b7x + <span class='b-color'>{fmt(b)}</span>"
        )
        el(f"point-out-{label}").innerHTML = (
            f"pred = {fmt(pred)} <span class='target-note'>(target {fmt(y)})</span>"
        )


def render_error_row(err):
    el("error-node").textContent = f"E ({error_label()}) = {fmt(err['E'])}"


def render_grad_row(err):
    el("grad-w-node").textContent = f"dE/dw = {fmt(err['dE_dw'])}"
    el("grad-b-node").textContent = f"dE/db = {fmt(err['dE_db'])}"


def render_update_row(upd):
    el("update-w-node").textContent = f"\u0394w = {fmt(upd['delta_w'])}"
    el("update-b-node").textContent = f"\u0394b = {fmt(upd['delta_b'])}"


def render_final_row(upd):
    final_w = el("final-w-node")
    final_w.innerHTML = (
        f"<span class='dir-arrow dir-{upd['w_dir']}'>{ARROW_GLYPH[upd['w_dir']]}</span> "
        f"w = {fmt(upd['w_new'])}"
    )
    final_b = el("final-b-node")
    final_b.innerHTML = (
        f"<span class='dir-arrow dir-{upd['b_dir']}'>{ARROW_GLYPH[upd['b_dir']]}</span> "
        f"b = {fmt(upd['b_new'])}"
    )


def render_live_network_diagram(w, b, highlight=False):
    w_dir = state.last_w_dir
    b_dir = state.last_b_dir
    w_arrow = f"<span class='dir-arrow dir-{w_dir}'>{ARROW_GLYPH[w_dir]}</span> " if w_dir else ""
    b_arrow = f"<span class='dir-arrow dir-{b_dir}'>{ARROW_GLYPH[b_dir]}</span> " if b_dir else ""
    live_network_eq_el.innerHTML = (
        f"{w_arrow}<span class='w-color'>{fmt(w)}</span>\u00b7x + "
        f"{b_arrow}<span class='b-color'>{fmt(b)}</span>"
    )
    if highlight:
        live_network_eq_el.classList.add("highlight-pulse")
    else:
        live_network_eq_el.classList.remove("highlight-pulse")


def reset_flow_display():
    """Blank the flowchart back to placeholders (used for the very
    first, pre-training snapshot)."""
    lin_html = f"<span class='w-color'>{fmt(state.w)}</span>\u00b7x + <span class='b-color'>{fmt(state.b)}</span>"
    el("point-x-p1").textContent = "x = –"
    el("point-x-p2").textContent = "x = –"
    el("point-lin-p1").innerHTML = lin_html
    el("point-lin-p2").innerHTML = lin_html
    el("point-out-p1").innerHTML = "pred = – <span class='target-note'>(target –)</span>"
    el("point-out-p2").innerHTML = "pred = – <span class='target-note'>(target –)</span>"
    el("error-node").textContent = "E = –"
    el("grad-w-node").textContent = "dE/dw = –"
    el("grad-b-node").textContent = "dE/db = –"
    el("update-w-node").textContent = "\u0394w = –"
    el("update-b-node").textContent = "\u0394b = –"
    el("final-w-node").innerHTML = "<span class='dir-arrow'>–</span> w = –"
    el("final-b-node").innerHTML = "<span class='dir-arrow'>–</span> b = –"
    set_all_states("inactive", "inactive", "inactive", "inactive", "inactive")
    render_slice_plots()


# ─────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────

def _forward_only(x, w, b):
    return w * x + b


def init_prediction_plot():
    x1, y1 = state.p1
    x2, y2 = state.p2
    lo = min(x1, x2) - 1
    hi = max(x1, x2) + 1
    xs = [lo + i * (hi - lo) / 40 for i in range(41)]
    ys = [_forward_only(x, state.w, state.b) for x in xs]

    line_trace = {"x": xs, "y": ys, "mode": "lines", "name": "prediction",
                  "line": {"color": "#3b6ff5", "width": 3}}
    points_trace = {"x": [x1, x2], "y": [y1, y2], "mode": "markers", "name": "points",
                     "marker": {"color": "#059669", "size": 10}}

    layout = {
        "margin": {"l": 36, "r": 14, "t": 6, "b": 26},
        "showlegend": False,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis": {"zeroline": True},
        "yaxis": {"zeroline": True},
        "font": {"size": 10},
    }
    Plotly.newPlot(prediction_plot_el, to_js([line_trace, points_trace]), to_js(layout),
                    to_js({"responsive": True, "displayModeBar": False}))


def update_prediction_plot():
    x1, y1 = state.p1
    x2, y2 = state.p2
    lo = min(x1, x2) - 1
    hi = max(x1, x2) + 1
    xs = [lo + i * (hi - lo) / 40 for i in range(41)]
    ys = [_forward_only(x, state.w, state.b) for x in xs]
    Plotly.restyle(
        prediction_plot_el,
        to_js({"x": [xs, [x1, x2]], "y": [ys, [y1, y2]]}),
        to_js([0, 1]),
    )


def init_loss_plot():
    layout = {
        "margin": {"l": 44, "r": 14, "t": 10, "b": 34},
        "showlegend": False,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"size": 10},
        "xaxis": {"title": {"text": "Epoch", "font": {"size": 10}}},
        "yaxis": {"title": {"text": "Loss", "font": {"size": 10}}},
    }
    trace = {"x": [], "y": [], "mode": "lines+markers", "line": {"color": "#d97706", "width": 2}}
    Plotly.newPlot(loss_plot_el, to_js([trace]), to_js(layout),
                    to_js({"responsive": True, "displayModeBar": False}))


def update_loss_plot():
    xs = [e for e, _ in state.loss_history]
    ys = [v for _, v in state.loss_history]
    Plotly.restyle(loss_plot_el, to_js({"x": [xs], "y": [ys]}), to_js([0]))


# ── Axis-aligned loss slices: E vs w (b fixed) and E vs b (w fixed) ──
#
# Unlike a slice along the raw gradient direction, these two slices
# each have a curvature (second derivative) that does NOT change as
# training proceeds and the gradient's direction rotates. That means
# they stay honest parabolas (or V-shapes for MAE) of consistent
# steepness the whole time, and the slope at t=0 IS exactly the
# signed partial derivative shown in the flowchart above them - no
# arrow-direction ambiguity needed.

def _slice_base_layout(param_symbol):
    return {
        "margin": {"l": 34, "r": 10, "t": 20, "b": 26},
        "showlegend": False,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"size": 9},
        "xaxis": {
            "title": {"text": param_symbol, "font": {"size": 9}},
            "zeroline": False,
        },
        "yaxis": {"title": {"text": "E", "font": {"size": 8}}},
    }


def _placeholder_layout(param_symbol):
    layout = _slice_base_layout(param_symbol)
    layout["xaxis"]["range"] = [-1, 1]
    layout["yaxis"]["range"] = [0, 1]
    layout["annotations"] = [{
        "text": "Step forward to compute \u2207E",
        "x": 0.5, "y": 0.5, "xref": "paper", "yref": "paper",
        "showarrow": False, "font": {"size": 10, "color": "#9ca3b8"},
    }]
    return layout


def init_w_slice_plot():
    Plotly.newPlot(w_slice_plot_el, to_js([]), to_js(_placeholder_layout("w")),
                    to_js({"responsive": True, "displayModeBar": False}))


def init_b_slice_plot():
    Plotly.newPlot(b_slice_plot_el, to_js([]), to_js(_placeholder_layout("b")),
                    to_js({"responsive": True, "displayModeBar": False}))


def _eval_E_at(w_val, b_val, e_fn):
    x1, y1 = state.p1
    x2, y2 = state.p2
    try:
        pred1 = w_val * x1 + b_val
        pred2 = w_val * x2 + b_val
        return (e_fn(pred1, y1) + e_fn(pred2, y2)) / 2
    except Exception:
        return None


def _find_slice_window(eval_t, initial_T):
    """Grow T geometrically until both edges of [-T, T] are clearly
    higher than the center AND higher than their own midpoints - i.e.
    the curve has visibly turned back upward on both sides, so the
    minimum (or at least a full 'bounce') is always inside the frame."""
    T = initial_T if initial_T > 1e-6 else 0.5
    e0 = eval_t(0.0)
    if e0 is None:
        return T
    for _ in range(50):
        eL, eR = eval_t(-T), eval_t(T)
        eLm, eRm = eval_t(-T / 2), eval_t(T / 2)
        if None not in (eL, eR, eLm, eRm) and eL > e0 and eR > e0 and eL >= eLm and eR >= eRm:
            return T
        T *= 1.6
        if T > 1e6:
            break
    return T


def compute_axis_slice(fixed_val, moving_val0, is_w, e_fn):
    def eval_t(t):
        if is_w:
            return _eval_E_at(moving_val0 + t, fixed_val, e_fn)
        return _eval_E_at(fixed_val, moving_val0 + t, e_fn)

    init_T = max(abs(state.lr) * 4, 0.5)
    # Find a window guaranteed to show the curve turning back upward on
    # both sides, then pull it in a bit - still centered on the current
    # value, still shows the parabola shape, just a tighter crop.
    T_full = _find_slice_window(eval_t, init_T)
    T = T_full * 0.7
    n = 61
    ts = [-T + i * (2 * T) / (n - 1) for i in range(n)]
    xs = [moving_val0 + t for t in ts]  # actual w (or b) values, not offsets
    es = [eval_t(t) for t in ts]
    return xs, es, T, moving_val0


def _render_one_slice(plot_el, xs, es, T, x0, E0, grad_val, color, param_symbol):
    valid_pairs = [(x, e) for x, e in zip(xs, es) if e is not None]
    valid = [e for _, e in valid_pairs]
    y_lo = min(valid + [E0])
    y_hi = max(valid + [E0])
    pad = (y_hi - y_lo) * 0.12
    if pad <= 0:
        pad = max(abs(E0) * 0.12, 0.1)
    y_lo -= pad
    y_hi += pad

    # Locate the slice's own vertex (its minimum, numerically, from the
    # sampled curve) - this is the SECOND dividing line. Between the
    # current value and the vertex, loss is genuinely decreasing; past
    # the vertex on either side, it's increasing again, since this is a
    # parabola (or V-shape).
    if valid_pairs:
        x_vertex, _ = min(valid_pairs, key=lambda pair: pair[1])
    else:
        x_vertex = x0
    x_vertex = max(x0 - T, min(x0 + T, x_vertex))

    curve_trace = {
        "x": xs, "y": es, "mode": "lines",
        "line": {"color": color, "width": 2.5},
        "hovertemplate": f"{param_symbol}=%{{x:.3f}}<br>E=%{{y:.4f}}<extra></extra>",
    }
    point_trace = {
        "x": [x0], "y": [E0], "mode": "markers",
        "marker": {"color": "#1a1d2e", "size": 8, "line": {"color": "#fff", "width": 1.5}},
    }

    shapes = [
        {"type": "line", "xref": "x", "yref": "paper", "x0": x0, "x1": x0, "y0": 0, "y1": 1,
         "line": {"color": "#9ca3b8", "width": 1, "dash": "dot"}},
        {"type": "line", "xref": "x", "yref": "paper", "x0": x_vertex, "x1": x_vertex, "y0": 0, "y1": 1,
         "line": {"color": "#9ca3b8", "width": 1, "dash": "dot"}},
    ]
    annotations = [{
        "x": x0, "y": E0, "ax": 0, "ay": -26, "xref": "x", "yref": "y",
        "text": f"dE/d{param_symbol} = {fmt(grad_val)}",
        "showarrow": True, "arrowcolor": color, "arrowwidth": 2, "arrowhead": 2,
        "font": {"size": 9, "color": color}, "bgcolor": "rgba(255,255,255,0.85)",
        "bordercolor": color, "borderwidth": 1, "borderpad": 2,
    }]

    if abs(x_vertex - x0) > 1e-9:
        lo, hi = (x0, x_vertex) if x_vertex > x0 else (x_vertex, x0)
        shapes.append({"type": "rect", "xref": "x", "yref": "paper",
                        "x0": lo, "x1": hi, "y0": 0, "y1": 1,
                        "fillcolor": "rgba(5,150,105,0.09)", "line": {"width": 0}, "layer": "below"})
        for seg_lo, seg_hi in ((x0 - T, min(x0, x_vertex)), (max(x0, x_vertex), x0 + T)):
            if seg_hi - seg_lo > 1e-9:
                shapes.append({"type": "rect", "xref": "x", "yref": "paper",
                                "x0": seg_lo, "x1": seg_hi, "y0": 0, "y1": 1,
                                "fillcolor": "rgba(220,38,38,0.07)", "line": {"width": 0}, "layer": "below"})
        annotations.append({
            "x": x_vertex, "y": 0, "yref": "paper", "yanchor": "bottom", "xref": "x",
            "text": "min", "showarrow": False, "font": {"size": 8, "color": "#9ca3b8"},
        })

    layout = _slice_base_layout(param_symbol)
    layout["shapes"] = shapes
    layout["annotations"] = annotations
    layout["xaxis"]["range"] = [x0 - T, x0 + T]
    layout["yaxis"]["range"] = [y_lo, y_hi]

    Plotly.react(plot_el, to_js([curve_trace, point_trace]), to_js(layout),
                 to_js({"responsive": True, "displayModeBar": False}))


def render_slice_plots(w0=None, b0=None, grad=None):
    """Render both the E-vs-w and E-vs-b slices around the given point.
    Grays the panels out (and shows placeholders) until a gradient has
    actually been computed for this step."""
    if grad is None:
        grad = state.grad_result
    if w0 is None:
        w0 = state.w
    if b0 is None:
        b0 = state.b

    if grad is None:
        w_slice_panel_el.classList.add("slice-plot-inactive")
        b_slice_panel_el.classList.add("slice-plot-inactive")
        init_w_slice_plot()
        init_b_slice_plot()
        return

    try:
        e_fn, _ = resolve_error_fns()
    except Exception:
        return

    E0 = grad["E"]
    dE_dw, dE_db = grad["dE_dw"], grad["dE_db"]

    xs_w, es_w, T_w, w_x0 = compute_axis_slice(b0, w0, True, e_fn)
    _render_one_slice(w_slice_plot_el, xs_w, es_w, T_w, w_x0, E0, dE_dw, "#0e6b7a", "w")

    xs_b, es_b, T_b, b_x0 = compute_axis_slice(w0, b0, False, e_fn)
    _render_one_slice(b_slice_plot_el, xs_b, es_b, T_b, b_x0, E0, dE_db, "#b45309", "b")

    w_slice_panel_el.classList.remove("slice-plot-inactive")
    b_slice_panel_el.classList.remove("slice-plot-inactive")


def resize_plots():
    try:
        Plotly.Plots.resize(prediction_plot_el)
        Plotly.Plots.resize(loss_plot_el)
        Plotly.Plots.resize(w_slice_plot_el)
        Plotly.Plots.resize(b_slice_plot_el)
    except Exception:
        pass


async def resize_plots_soon():
    await asyncio.sleep(0.05)
    resize_plots()
    await asyncio.sleep(0.3)
    resize_plots()


# ─────────────────────────────────────────────────────────────────
# Snapshotting for backward step/epoch
# ─────────────────────────────────────────────────────────────────

def take_snapshot():
    state.history.append({
        "epoch": state.epoch,
        "step_index": state.step_index,
        "w": state.w,
        "b": state.b,
        "forward_result": dict(state.forward_result) if state.forward_result else None,
        "grad_result": dict(state.grad_result) if state.grad_result else None,
        "update_result": dict(state.update_result) if state.update_result else None,
        "loss_history": list(state.loss_history),
        "last_w_dir": state.last_w_dir,
        "last_b_dir": state.last_b_dir,
    })


def restore_snapshot(snap):
    state.epoch = snap["epoch"]
    state.step_index = snap["step_index"]
    state.w = snap["w"]
    state.b = snap["b"]
    state.forward_result = dict(snap["forward_result"]) if snap["forward_result"] else None
    state.grad_result = dict(snap["grad_result"]) if snap["grad_result"] else None
    state.update_result = dict(snap["update_result"]) if snap["update_result"] else None
    state.loss_history = list(snap["loss_history"])
    state.last_w_dir = snap.get("last_w_dir")
    state.last_b_dir = snap.get("last_b_dir")

    if state.forward_result is None:
        reset_flow_display()
    else:
        render_point_diagrams(state.forward_result, state.w, state.b)
        render_error_row(state.grad_result)

        if state.step_index == 1:
            # grad/update/final not computed yet for this epoch
            set_all_states("active", "active", "inactive", "inactive", "inactive")
        elif state.step_index == 2:
            render_grad_row(state.grad_result)
            set_all_states("done", "done", "active", "inactive", "inactive")
        elif state.step_index == 3:
            render_grad_row(state.grad_result)
            render_update_row(state.update_result)
            set_all_states("done", "done", "done", "active", "inactive")
        else:
            # step_index == 0: epoch just completed, everything computed
            render_grad_row(state.grad_result)
            render_update_row(state.update_result)
            render_final_row(state.update_result)
            set_all_states("done", "done", "done", "done", "active")

    highlight = state.step_index == 0 and state.forward_result is not None
    render_live_network_diagram(state.w, state.b, highlight=highlight)
    update_prediction_plot()
    update_loss_plot()
    render_slice_plots()
    update_back_button_states()


def update_back_button_states():
    can_go_back = len(state.history) > 1
    back_step_btn.disabled = not can_go_back
    back_epoch_btn.disabled = not can_go_back


# ─────────────────────────────────────────────────────────────────
# Init / randomize
# ─────────────────────────────────────────────────────────────────

def read_setup_inputs():
    try:
        state.p1 = (float(p1x_input.value), float(p1y_input.value))
        state.p2 = (float(p2x_input.value), float(p2y_input.value))
    except ValueError:
        return False

    state.error_key = error_select.value
    state.custom_error_expr = custom_error_input.value

    try:
        state.lr = float(lr_input.value)
    except ValueError:
        return False

    return True


def randomize_and_reset():
    if not read_setup_inputs():
        return False

    state.w = random.uniform(*INIT_RANGE)
    state.b = random.uniform(*INIT_RANGE)

    state.epoch = 0
    state.step_index = 0
    state.loss_history = []
    state.forward_result = None
    state.grad_result = None
    state.update_result = None
    state.initialized = True
    state.history = []
    state.last_w_dir = None
    state.last_b_dir = None

    build_flow_skeleton()
    reset_flow_display()
    render_live_network_diagram(state.w, state.b)
    init_prediction_plot()
    init_loss_plot()
    init_w_slice_plot()
    init_b_slice_plot()
    enable_training_controls(True)
    take_snapshot()
    update_back_button_states()
    return True


def ensure_initialized():
    if not state.initialized:
        return randomize_and_reset()
    return True


def on_randomize_click(event=None):
    was_playing = state.playing
    if was_playing:
        stop_playing()
    randomize_and_reset()


# ─────────────────────────────────────────────────────────────────
# Step state machine
# ─────────────────────────────────────────────────────────────────

def do_step():
    if not ensure_initialized():
        return

    set_setup_controls_locked(True)

    if state.step_index == 0:
        try:
            e_fn, grad_fn = resolve_error_fns()
        except Exception:
            return

        fwd = forward(state.p1[0], state.p1[1], state.p2[0], state.p2[1], state.w, state.b)
        state.forward_result = fwd
        render_point_diagrams(fwd, state.w, state.b)
        render_live_network_diagram(state.w, state.b, highlight=False)

        try:
            err = compute_error_and_grads(
                state.p1[0], state.p1[1], fwd["pred1"],
                state.p2[0], state.p2[1], fwd["pred2"],
                e_fn, grad_fn,
            )
        except Exception:
            return

        state.grad_result = err
        state.update_result = None
        render_error_row(err)
        render_slice_plots()
        set_all_states("active", "active", "inactive", "inactive", "inactive")
        state.step_index = 1

    elif state.step_index == 1:
        render_grad_row(state.grad_result)
        set_all_states("done", "done", "active", "inactive", "inactive")
        state.step_index = 2

    elif state.step_index == 2:
        upd = compute_update(state.w, state.b, state.grad_result["dE_dw"], state.grad_result["dE_db"], state.lr)
        state.update_result = upd
        render_update_row(upd)
        set_all_states("done", "done", "done", "active", "inactive")
        state.step_index = 3

    else:
        render_final_row(state.update_result)
        set_all_states("done", "done", "done", "done", "active")
        state.last_w_dir = state.update_result["w_dir"]
        state.last_b_dir = state.update_result["b_dir"]
        render_live_network_diagram(state.update_result["w_new"], state.update_result["b_new"], highlight=True)

        state.loss_history.append((state.epoch, state.grad_result["E"]))
        update_loss_plot()

        state.w, state.b = state.update_result["w_new"], state.update_result["b_new"]
        update_prediction_plot()

        state.epoch += 1
        state.step_index = 0

    take_snapshot()
    update_back_button_states()


def run_epoch():
    if not ensure_initialized():
        return
    do_step()
    while state.step_index != 0:
        do_step()


def run_epoch_turbo():
    """Used while Play is running: does the full epoch's math in one
    shot and renders the result once, instead of once per sub-step.
    This keeps Play both faster and less visually busy."""
    if not ensure_initialized():
        return

    try:
        e_fn, grad_fn = resolve_error_fns()
    except Exception:
        return

    w_used, b_used = state.w, state.b
    fwd = forward(state.p1[0], state.p1[1], state.p2[0], state.p2[1], w_used, b_used)

    try:
        err = compute_error_and_grads(
            state.p1[0], state.p1[1], fwd["pred1"],
            state.p2[0], state.p2[1], fwd["pred2"],
            e_fn, grad_fn,
        )
    except Exception:
        return

    upd = compute_update(w_used, b_used, err["dE_dw"], err["dE_db"], state.lr)

    state.forward_result = fwd
    state.grad_result = err
    state.update_result = upd
    state.loss_history.append((state.epoch, err["E"]))
    state.last_w_dir = upd["w_dir"]
    state.last_b_dir = upd["b_dir"]
    state.w, state.b = upd["w_new"], upd["b_new"]
    state.epoch += 1
    state.step_index = 0

    render_point_diagrams(fwd, w_used, b_used)
    render_error_row(err)
    render_grad_row(err)
    render_update_row(upd)
    render_final_row(upd)
    set_all_states("done", "done", "done", "done", "done")
    render_live_network_diagram(state.w, state.b, highlight=True)
    update_prediction_plot()
    update_loss_plot()
    render_slice_plots(w_used, b_used, err)

    take_snapshot()
    update_back_button_states()


def do_backward_step():
    if len(state.history) <= 1:
        return
    state.history.pop()
    snap = state.history[-1]
    restore_snapshot(snap)


def backward_epoch():
    if len(state.history) <= 1:
        return
    do_backward_step()
    while state.step_index != 0 and len(state.history) > 1:
        do_backward_step()


# ─────────────────────────────────────────────────────────────────
# Play / Pause
# ─────────────────────────────────────────────────────────────────

PLAY_TURBO_DELAY = 0.08  # faster than manual stepping; updates once per epoch

_play_task = None


async def play_loop():
    while state.playing:
        run_epoch_turbo()
        await asyncio.sleep(PLAY_TURBO_DELAY)


def enable_training_controls(enabled):
    step_btn.disabled = not enabled
    epoch_btn.disabled = not enabled


def set_setup_controls_locked(locked):
    if locked:
        setup_row_el.classList.add("controls-locked")
    else:
        setup_row_el.classList.remove("controls-locked")
    for field in (p1x_input, p1y_input, p2x_input, p2y_input,
                  error_select, custom_error_input, lr_input, randomize_btn):
        field.disabled = locked


def start_playing():
    global _play_task
    if not ensure_initialized():
        return
    state.playing = True
    play_pause_btn.textContent = "❚❚"
    play_pause_btn.classList.add("is-playing")
    play_pause_btn.title = "Pause"
    set_setup_controls_locked(True)
    _play_task = asyncio.ensure_future(play_loop())


def stop_playing():
    state.playing = False
    play_pause_btn.textContent = "\u25b6"
    play_pause_btn.classList.remove("is-playing")
    play_pause_btn.title = "Play"


def on_play_pause_click(event=None):
    if state.playing:
        stop_playing()
    else:
        start_playing()


def on_reset_click(event=None):
    if state.playing:
        stop_playing()
    set_setup_controls_locked(False)
    randomize_and_reset()


def on_back_step_click(event=None):
    if state.playing:
        stop_playing()
    do_backward_step()


def on_back_epoch_click(event=None):
    if state.playing:
        stop_playing()
    backward_epoch()


def on_point_input_change(event=None):
    try:
        new_p1 = (float(p1x_input.value), float(p1y_input.value))
        new_p2 = (float(p2x_input.value), float(p2y_input.value))
    except ValueError:
        return

    state.p1 = new_p1
    state.p2 = new_p2

    if not state.initialized:
        return

    update_prediction_plot()

    if state.forward_result is not None:
        x1, y1 = state.p1
        x2, y2 = state.p2
        el("point-x-p1").textContent = f"x = {fmt(x1)}"
        el("point-x-p2").textContent = f"x = {fmt(x2)}"
        out1 = el("point-out-p1")
        out2 = el("point-out-p2")
        pred1_text = out1.querySelector(".target-note") if out1 else None
        pred2_text = out2.querySelector(".target-note") if out2 else None
        if pred1_text is not None:
            pred1_text.textContent = f"(target {fmt(y1)})"
        if pred2_text is not None:
            pred2_text.textContent = f"(target {fmt(y2)})"


def on_error_select_change(event=None):
    if error_select.value == "custom":
        custom_error_input.classList.remove("hidden")
    else:
        custom_error_input.classList.add("hidden")


def on_toggle_explanation_click(event=None):
    flowchart_panel_el.classList.toggle("hidden")
    if flowchart_panel_el.classList.contains("hidden"):
        toggle_explanation_btn.textContent = "Show Explanation"
    else:
        toggle_explanation_btn.textContent = "Hide Explanation"


# ─────────────────────────────────────────────────────────────────
# Event wiring
# ─────────────────────────────────────────────────────────────────

randomize_btn.addEventListener("click", create_proxy(on_randomize_click))
step_btn.addEventListener("click", create_proxy(lambda e: do_step()))
epoch_btn.addEventListener("click", create_proxy(lambda e: run_epoch()))
back_step_btn.addEventListener("click", create_proxy(on_back_step_click))
back_epoch_btn.addEventListener("click", create_proxy(on_back_epoch_click))
play_pause_btn.addEventListener("click", create_proxy(on_play_pause_click))
reset_btn.addEventListener("click", create_proxy(on_reset_click))
error_select.addEventListener("change", create_proxy(on_error_select_change))
toggle_explanation_btn.addEventListener("click", create_proxy(on_toggle_explanation_click))
point_input_proxy = create_proxy(on_point_input_change)
p1x_input.addEventListener("input", point_input_proxy)
p1y_input.addEventListener("input", point_input_proxy)
p2x_input.addEventListener("input", point_input_proxy)
p2y_input.addEventListener("input", point_input_proxy)
window_resize_proxy = create_proxy(lambda e: resize_plots())
document.defaultView.addEventListener("resize", window_resize_proxy)

# ─────────────────────────────────────────────────────────────────
# Boot — everything is live immediately, no button press required
# ─────────────────────────────────────────────────────────────────

on_error_select_change()
build_flow_skeleton()
enable_training_controls(False)
randomize_and_reset()
el("loading-splash").classList.add("hidden")
el("app").classList.remove("hidden")
asyncio.ensure_future(resize_plots_soon())