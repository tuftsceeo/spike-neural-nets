"""The step machine: forward pass -> two backward-reveal sub-steps per
layer (activation, then linear/weight; right to left) -> epoch complete.
Mirrors gradient-descent-visualization's do_step()/run_epoch()/play_loop()
pattern, generalized to N layers and to two chain-rule stages per layer."""
import asyncio

import state
import network_model
import diagram_render
import plots

step_btn = state.get_id("step-btn")
epoch_btn = state.get_id("epoch-btn")
back_step_btn = state.get_id("back-step-btn")
back_epoch_btn = state.get_id("back-epoch-btn")
play_pause_btn = state.get_id("play-pause-btn")


def ensure_initialized() -> bool:
    return bool(state.layers) and bool(state.dataset)


def update_back_button_states():
    can_go_back = len(state.history) > 1
    back_step_btn.disabled = not can_go_back
    back_epoch_btn.disabled = not can_go_back


def _total_substeps() -> int:
    return 2 * len(state.plan)


def do_step():
    if not ensure_initialized():
        return

    if state.step_index == 0:
        state.forward_cache = network_model.compute_forward_cache()
        state.plan = network_model.build_backward_plan()
        diagram_render.clear_grad_markers()
        diagram_render.clear_all_highlights()
        diagram_render.render_output_readout()
        state.step_index = 1
    else:
        plan_idx = (state.step_index - 1) // 2
        entry = state.plan[plan_idx]
        is_activation = state.step_index % 2 == 1

        if is_activation:
            diagram_render.render_activation_reveal(entry)
        else:
            network_model.apply_plan_step(plan_idx + 1)
            diagram_render.render_linear_reveal(entry)
            plots.update_fit_curve()

        if state.step_index >= _total_substeps():
            state.loss_history.append((state.epoch, state.forward_cache["mean_loss"]))
            plots.update_loss_plot()
            state.epoch += 1
            state.step_index = 0
        else:
            state.step_index += 1

    network_model.take_snapshot()
    update_back_button_states()


def run_epoch():
    if not ensure_initialized():
        return
    do_step()
    while state.step_index != 0:
        do_step()


def run_epoch_turbo():
    """Used while Play is running: does the whole epoch's math and applies
    every layer's update in one shot, rendering the result once instead of
    once per sub-step -- faster and less visually busy than stepping."""
    if not ensure_initialized():
        return

    state.forward_cache = network_model.compute_forward_cache()
    state.plan = network_model.build_backward_plan()
    for i in range(1, len(state.plan) + 1):
        network_model.apply_plan_step(i)

    diagram_render.clear_grad_markers()
    diagram_render.clear_all_highlights()
    diagram_render.render_output_readout()
    diagram_render.render_weight_badges()
    plots.update_fit_curve()

    state.loss_history.append((state.epoch, state.forward_cache["mean_loss"]))
    plots.update_loss_plot()
    state.epoch += 1
    state.step_index = 0

    network_model.take_snapshot()
    update_back_button_states()


def _replay_revealed_substeps():
    """Rebuilds every arrow/label/highlight up through the CURRENT
    step_index -- used after restoring a snapshot, since the snapshot only
    carries raw state, not the visual markers."""
    diagram_render.clear_grad_markers()
    diagram_render.clear_all_highlights()
    if not state.plan or state.step_index < 1:
        return
    for s in range(1, state.step_index + 1):
        plan_idx = (s - 1) // 2
        entry = state.plan[plan_idx]
        if s % 2 == 1:
            diagram_render.render_activation_reveal(entry)
        else:
            diagram_render.render_linear_reveal(entry)


def do_backward_step():
    if len(state.history) <= 1:
        return
    state.history.pop()
    snap = state.history[-1]
    network_model.restore_snapshot(snap)
    diagram_render.render_weight_badges()
    diagram_render.render_output_readout()
    _replay_revealed_substeps()
    plots.update_fit_curve()
    plots.update_loss_plot()
    update_back_button_states()


def backward_epoch():
    if len(state.history) <= 1:
        return
    do_backward_step()
    while state.step_index != 0 and len(state.history) > 1:
        do_backward_step()


# ── Play / Pause ────────────────────────────────────────────────────────

PLAY_DELAY = 0.03
_play_task = None


async def play_loop():
    while state.playing:
        run_epoch_turbo()
        await asyncio.sleep(PLAY_DELAY)


def enable_training_controls(enabled: bool):
    step_btn.disabled = not enabled
    epoch_btn.disabled = not enabled


def start_playing():
    global _play_task
    if not ensure_initialized():
        return
    state.playing = True
    play_pause_btn.textContent = "❚❚"
    play_pause_btn.classList.add("is-playing")
    play_pause_btn.title = "Pause"
    _play_task = asyncio.ensure_future(play_loop())


def stop_playing():
    state.playing = False
    play_pause_btn.textContent = "▶"
    play_pause_btn.classList.remove("is-playing")
    play_pause_btn.title = "Play"


def on_play_pause_click(evt=None):
    if state.playing:
        stop_playing()
    else:
        start_playing()


def on_step_click(evt=None):
    if state.playing:
        stop_playing()
    do_step()


def on_epoch_click(evt=None):
    if state.playing:
        stop_playing()
    run_epoch()


def on_back_step_click(evt=None):
    if state.playing:
        stop_playing()
    do_backward_step()


def on_back_epoch_click(evt=None):
    if state.playing:
        stop_playing()
    backward_epoch()
