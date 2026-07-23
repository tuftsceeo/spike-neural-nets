import json

from js import document, JSON


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
setup_row_el = el("controls-cell")
dataset_section_el = el("dataset-section")
back_epoch_btn = el("back-epoch-btn")
back_step_btn = el("back-step-btn")
step_btn = el("step-btn")
epoch_btn = el("epoch-btn")
play_pause_btn = el("play-pause-btn")
toggle_explanation_btn = el("toggle-explanation-btn")
flowchart_panel_el = el("flowchart-panel")
point_pass_col_el = el("point-pass-stack")
prediction_plot_el = el("prediction-plot")
loss_plot_el = el("loss-plot")
live_network_eq_el = el("live-network-eq")

w_slice_panel_el = el("w-slice-panel")
b_slice_panel_el = el("b-slice-panel")
w_slice_plot_el = el("w-slice-plot")
b_slice_plot_el = el("b-slice-plot")
