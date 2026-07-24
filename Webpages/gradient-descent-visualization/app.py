import asyncio
import random

from pyodide.ffi import create_proxy
from js import document

from state import state, INIT_RANGE
from dom import (
    el, p1x_input, p1y_input, p2x_input, p2y_input, error_select,
    custom_error_input, lr_input, randomize_btn, reset_btn, setup_row_el,
    dataset_section_el, back_epoch_btn, back_step_btn, step_btn, epoch_btn,
    play_pause_btn, toggle_explanation_btn, flowchart_panel_el,
)
from math_core import resolve_error_fns, compute_error_and_grads, forward, \
    compute_update, fmt
from flowchart import (
    build_flow_skeleton, build_point_pass_skeleton, set_all_states,
    render_point_diagrams, render_error_row, render_grad_row,
    render_update_row, render_final_row, render_live_network_diagram,
    reset_flow_display,
)
from plots import (
    init_prediction_plot, update_prediction_plot, init_loss_plot,
    update_loss_plot, init_w_slice_plot, init_b_slice_plot,
    render_slice_plots, resize_plots, resize_plots_soon,
)

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
            render_slice_plots()
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
    build_point_pass_skeleton()
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

async def _float_value_to(text, color, source_el, dest_el, src_x_frac=0.5, dest_x_frac=0.5):
    """Animates a small floating clone of `text` from a point within
    source_el to a point within dest_el, at full size/opacity the whole
    way, then simply disappears on arrival.

    src_x_frac / dest_x_frac control how far across the source's /
    destination's width the start/end point sits (0 = left edge,
    1 = right edge, 0.5 = center)."""
    if source_el is None or dest_el is None:
        return
    try:
        src_rect = source_el.getBoundingClientRect()
        dst_rect = dest_el.getBoundingClientRect()
    except Exception:
        return

    src_x = src_rect.left + src_rect.width * src_x_frac
    src_y = src_rect.top + src_rect.height / 2

    clone = document.createElement("div")
    clone.className = "float-up-value"
    clone.textContent = text
    clone.style.left = f"{src_x}px"
    clone.style.top = f"{src_y}px"
    clone.style.color = color
    document.body.appendChild(clone)

    # Force layout so the starting position is committed before we
    # change it - otherwise the browser may skip straight to the final
    # position with no visible transition.
    _ = clone.offsetHeight

    dst_x = dst_rect.left + dst_rect.width * dest_x_frac
    dst_y = dst_rect.top + dst_rect.height / 2
    dx = dst_x - src_x
    dy = dst_y - src_y
    clone.style.transform = f"translate({dx}px, {dy}px)"

    await asyncio.sleep(0.6)
    clone.remove()


async def animate_values_replacing_network(w_new, b_new):
    """At the end of a manual epoch step, floats the final w/b values up
    from the flowchart's last row to the live network diagram - w
    heading toward the left side of the equation node (where the w
    term sits) and b toward the right side (where the b term sits).
    The network's displayed w/b only actually update once the floating
    values arrive - not before - so it reads as the new values
    traveling up and becoming the network's values, rather than the
    network instantly changing while numbers fly past it."""
    w_source = el("final-w-node")
    b_source = el("final-b-node")
    dest = el("live-network-eq")
    if w_source is None or b_source is None or dest is None:
        render_live_network_diagram(w_new, b_new, highlight=True)
        return

    await asyncio.gather(
        _float_value_to(f"w = {fmt(w_new)}", "#0e6b7a", w_source, dest,
                         src_x_frac=0.2, dest_x_frac=0.10),
        _float_value_to(f"b = {fmt(b_new)}", "#b45309", b_source, dest,
                         src_x_frac=0.2, dest_x_frac=0.62),
    )

    render_live_network_diagram(w_new, b_new, highlight=True)
    dest.classList.add("value-pop")
    await asyncio.sleep(0.3)
    dest.classList.remove("value-pop")


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
        set_all_states("active", "active", "inactive", "inactive", "inactive")
        state.step_index = 1

    elif state.step_index == 1:
        render_grad_row(state.grad_result)
        set_all_states("done", "done", "active", "inactive", "inactive")
        render_slice_plots()
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
        asyncio.ensure_future(animate_values_replacing_network(
            state.update_result["w_new"], state.update_result["b_new"]
        ))

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

PLAY_TURBO_DELAY = 0.04  # faster than manual stepping; updates once per epoch

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
        dataset_section_el.classList.add("controls-locked")
    else:
        setup_row_el.classList.remove("controls-locked")
        dataset_section_el.classList.remove("controls-locked")
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
    play_pause_btn.textContent = "▶"
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
build_point_pass_skeleton()
enable_training_controls(False)
randomize_and_reset()
el("loading-splash").classList.add("hidden")
el("app").classList.remove("hidden")
asyncio.ensure_future(resize_plots_soon())
