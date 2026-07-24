"""The two bottom-right/bottom-left plots: the current fit (data points +
the network's prediction curve) and the loss-vs-epoch trace."""
import json

from js import Plotly, JSON

import state
import network_model

fit_plot_el = state.get_id("fit-plot")
loss_plot_el = state.get_id("loss-plot")


def to_js(obj):
    return JSON.parse(json.dumps(obj))


def _x_range():
    if state.dataset:
        xs = [p["x"] for p in state.dataset]
        lo, hi = min(xs), max(xs)
        pad = max((hi - lo) * 0.15, 0.5)
        return lo - pad, hi + pad
    return -3.0, 3.0


def _curve_points():
    lo, hi = _x_range()
    xs = [lo + i * (hi - lo) / 60 for i in range(61)]
    ys = [network_model.predict(x) for x in xs]
    return xs, ys


def init_fit_plot():
    data_trace = {"x": [p["x"] for p in state.dataset], "y": [p["y"] for p in state.dataset],
                  "mode": "markers", "name": "data", "marker": {"color": "#059669", "size": 9}}
    xs, ys = _curve_points()
    curve_trace = {"x": xs, "y": ys, "mode": "lines", "name": "fit",
                   "line": {"color": "#3b6ff5", "width": 3}}
    layout = {
        "margin": {"l": 40, "r": 14, "t": 8, "b": 30},
        "showlegend": False,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"size": 10},
        "xaxis": {"zeroline": True},
        "yaxis": {"zeroline": True},
    }
    Plotly.newPlot(fit_plot_el, to_js([data_trace, curve_trace]), to_js(layout),
                    to_js({"responsive": True}))


def _rescale_fit_axes():
    """Plotly.restyle() alone doesn't always re-fit the axes to newly
    changed data -- once an axis has rendered, its range can stay "stuck"
    at the old extent. Force both axes to recompute so a newly added point
    outside the current view is actually visible. Only called when the
    DATASET changes (adding/removing a point is rare) -- NOT on every
    training step/epoch, since Play calls update_fit_curve() dozens of
    times a second and a relayout that often bogs the whole page down."""
    Plotly.relayout(fit_plot_el, to_js({"xaxis.autorange": True, "yaxis.autorange": True}))


def update_fit_data():
    Plotly.restyle(
        fit_plot_el,
        to_js({"x": [[p["x"] for p in state.dataset]], "y": [[p["y"] for p in state.dataset]]}),
        to_js([0]),
    )
    _rescale_fit_axes()


def update_fit_curve():
    xs, ys = _curve_points()
    Plotly.restyle(fit_plot_el, to_js({"x": [xs], "y": [ys]}), to_js([1]))


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
    trace = {"x": [], "y": [], "mode": "lines+markers", "line": {"color": "#d97706", "width": 2},
              "marker": {"size": 5}}
    Plotly.newPlot(loss_plot_el, to_js([trace]), to_js(layout),
                    to_js({"responsive": True}))


def update_loss_plot():
    xs = [e for e, _ in state.loss_history]
    ys = [v for _, v in state.loss_history]
    Plotly.restyle(loss_plot_el, to_js({"x": [xs], "y": [ys]}), to_js([0]))


def resize_plots():
    try:
        Plotly.Plots.resize(fit_plot_el)
        Plotly.Plots.resize(loss_plot_el)
    except Exception:
        pass
