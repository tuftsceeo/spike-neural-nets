import asyncio

from js import Plotly

from state import state
from dom import (
    to_js, prediction_plot_el, loss_plot_el, w_slice_plot_el, b_slice_plot_el,
    w_slice_panel_el, b_slice_panel_el,
)
from math_core import fmt, resolve_error_fns

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
        "text": "Step forward to compute ∇E",
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

    shapes = []
    annotations = [{
        "x": x0, "y": E0, "ax": 0, "ay": -26, "xref": "x", "yref": "y",
        "text": f"dE/d{param_symbol} = {fmt(grad_val)}",
        "showarrow": True, "arrowcolor": color, "arrowwidth": 2, "arrowhead": 2,
        "font": {"size": 9, "color": color}, "bgcolor": "rgba(255,255,255,0.85)",
        "bordercolor": color, "borderwidth": 1, "borderpad": 2,
    }]

    if abs(x_vertex - x0) > 1e-9:
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
