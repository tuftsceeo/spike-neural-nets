"""Small aggregator functions that re-render everything affected by a
given kind of change (topology, weights, dataset).

This lives in its own module -- NOT in main.py -- specifically so
dataset_ui.py and diagram_render.py can call it after a change. main.py
is the PyScript entry-point script (loaded via <script type="py"
src="main.py">), so it runs as __main__, not as a normally-cached module
named "main". A lazy `import main` from another module doesn't find it in
sys.modules under that name and instead re-fetches + re-executes the
whole file from scratch -- including boot() and wire_events() -- which
silently double-binds every button listener the first time it happens.
Importing this module instead avoids that trap entirely.
"""
import state
import network_model
import diagram_render
import plots
import training


def ready() -> bool:
    return bool(state.dataset) and bool(state.layers)


def refresh_plots_and_controls():
    plots.update_fit_data()
    plots.update_fit_curve()
    plots.update_loss_plot()
    training.enable_training_controls(ready())
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
