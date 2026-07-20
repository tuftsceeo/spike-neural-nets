"""Boot + event wiring for the horizontal backpropagator."""
import asyncio

from pyscript.ffi import create_proxy

import state
import network_model
import diagram_render
import dataset_ui
import plots
import training

add_layer_btn = state.get_id("add-layer-btn")
randomize_weights_btn = state.get_id("randomize-weights-btn")
biases_toggle = state.get_id("biases-toggle")
lr_input = state.get_id("lr-input")

step_btn = state.get_id("step-btn")
epoch_btn = state.get_id("epoch-btn")
back_step_btn = state.get_id("back-step-btn")
back_epoch_btn = state.get_id("back-epoch-btn")
play_pause_btn = state.get_id("play-pause-btn")
reset_btn = state.get_id("reset-btn")
close_act_help_btn = state.get_id("close-act-help-btn")


def _ready() -> bool:
    return bool(state.dataset) and bool(state.layers)


def refresh_plots_and_controls():
    plots.update_fit_data()
    plots.update_fit_curve()
    plots.update_loss_plot()
    training.enable_training_controls(_ready())
    network_model.take_snapshot()
    training.update_back_button_states()


def on_topology_changed():
    diagram_render.build_diagram()
    refresh_plots_and_controls()


def on_weight_change():
    diagram_render.render_weight_badges()
    diagram_render.clear_grad_markers()
    diagram_render.render_loss_readout()
    refresh_plots_and_controls()


def on_dataset_changed():
    diagram_render.clear_grad_markers()
    diagram_render.render_loss_readout()
    refresh_plots_and_controls()


# ── Top-panel controls ─────────────────────────────────────────────────────

def on_add_layer_click(evt=None):
    if state.playing:
        training.stop_playing()
    network_model.add_layer()
    on_topology_changed()


def on_randomize_weights_click(evt=None):
    if state.playing:
        training.stop_playing()
    network_model.randomize_weights()
    on_weight_change()


def on_biases_toggle_change(evt=None):
    if state.playing:
        training.stop_playing()
    network_model.set_biases_enabled(bool(evt.target.checked))
    on_topology_changed()


def on_lr_input_change(evt=None):
    try:
        state.lr = float(lr_input.value)
    except ValueError:
        pass


def on_reset_click(evt=None):
    """Same effect as Randomize weights, surfaced again next to Play so
    a learner mid-epoch can restart the whole run without reaching for
    the top-panel controls."""
    if state.playing:
        training.stop_playing()
    network_model.randomize_weights()
    on_weight_change()


# ── Boot ────────────────────────────────────────────────────────────────────

def seed_defaults():
    network_model.add_layer()
    for x, y in [(-2.0, -1.2), (-1.0, -0.4), (0.0, 0.2), (1.0, 0.9), (2.0, 1.6)]:
        network_model.add_data_point(x, y)


def wire_events():
    add_layer_btn.addEventListener("click", create_proxy(on_add_layer_click))
    randomize_weights_btn.addEventListener("click", create_proxy(on_randomize_weights_click))
    biases_toggle.addEventListener("change", create_proxy(on_biases_toggle_change))
    lr_input.addEventListener("input", create_proxy(on_lr_input_change))

    step_btn.addEventListener("click", create_proxy(training.on_step_click))
    epoch_btn.addEventListener("click", create_proxy(training.on_epoch_click))
    back_step_btn.addEventListener("click", create_proxy(training.on_back_step_click))
    back_epoch_btn.addEventListener("click", create_proxy(training.on_back_epoch_click))
    play_pause_btn.addEventListener("click", create_proxy(training.on_play_pause_click))
    reset_btn.addEventListener("click", create_proxy(on_reset_click))
    close_act_help_btn.addEventListener("click", create_proxy(diagram_render.close_act_help))

    def _on_resize(evt=None):
        plots.resize_plots()
        diagram_render.redraw_grad_markers()

    window_resize_proxy = create_proxy(_on_resize)
    from js import window
    window.addEventListener("resize", window_resize_proxy)


async def resize_plots_soon():
    await asyncio.sleep(0.05)
    plots.resize_plots()
    await asyncio.sleep(0.3)
    plots.resize_plots()


def boot():
    seed_defaults()

    diagram_render.build_diagram()
    dataset_ui.render_table()
    dataset_ui.render_add_row()
    plots.init_fit_plot()
    plots.init_loss_plot()

    training.enable_training_controls(_ready())
    network_model.take_snapshot()
    training.update_back_button_states()

    wire_events()

    state.get_id("loading-splash").classList.add("hidden")
    state.get_id("app").classList.remove("hidden")
    asyncio.ensure_future(resize_plots_soon())


boot()