"""Builds and updates the top network-diagram panel: the neuron chain, the
linear/weight and activation nodes, and the curved gradient arrows (with
their chain-rule labels) that appear under the diagram as backprop is
revealed one node at a time."""
import asyncio
import math

from pyscript import document
from pyscript.ffi import create_proxy

import state
import network_model
import ui_refresh

layers_track_el = state.get_id("layers-track")
loss_readout_el = state.get_id("loss-readout")
diagram_canvas_el = state.get_id("diagram-canvas")
grad_arrow_svg_el = state.get_id("grad-arrow-svg")
grad_label_layer_el = state.get_id("grad-label-layer")

SVG_NS = "http://www.w3.org/2000/svg"
ARROW_COLOR = "#7c3aed"


def make_el(tag, class_name=None, text=None, id_=None):
    e = document.createElement(tag)
    if class_name:
        e.className = class_name
    if text is not None:
        e.textContent = text
    if id_:
        e.id = id_
    return e


def fmt(x, n=3):
    try:
        return f"{x:.{n}f}"
    except (ValueError, TypeError):
        return str(x)


# ── Skeleton build (full rebuild whenever the topology changes) ──────────

def build_diagram():
    layers_track_el.innerHTML = ""
    for idx, layer in enumerate(state.layers):
        lid = layer["id"]
        pos = idx + 1
        source_name = "x" if idx == 0 else f"a{idx}"

        layers_track_el.appendChild(make_el("div", "flow-arrow", text="→"))

        conn_cell = make_el("div", "conn-cell")
        linear_title = make_el("div", "linear-title", text=f"n{pos}")
        linear_title.title = (
            f"n{pos} — the pre-activation value: {source_name} times the weight "
            f"(plus the bias, if on), before the activation function is applied."
        )
        conn_cell.appendChild(linear_title)
        weight_badge = make_el("div", "weight-badge", id_=f"weight-badge-{lid}")
        conn_cell.appendChild(weight_badge)
        layers_track_el.appendChild(conn_cell)

        # The linear node's output feeds the activation function -- show
        # that as its own arrow, not just adjacent boxes.
        layers_track_el.appendChild(make_el("div", "flow-arrow", text="→"))

        neuron_box = make_el("div", "neuron-box", id_=f"neuron-box-{lid}")

        title_row = make_el("div", "neuron-title-row")
        title_row.appendChild(make_el("div", "neuron-title", text=f"a{pos}"))
        help_btn = make_el("button", "act-help-btn", text="?")
        help_btn.title = "What do activation functions do?"
        title_row.appendChild(help_btn)
        neuron_box.appendChild(title_row)

        select = document.createElement("select")
        select.className = "act-select"
        select.id = f"act-select-{lid}"
        for label, value in state.ACTIVATION_OPTIONS:
            opt = document.createElement("option")
            opt.value = value
            opt.textContent = label
            if value == layer["act"]:
                opt.selected = True
            select.appendChild(opt)
        neuron_box.appendChild(select)

        remove_btn = make_el("button", "btn-remove-neuron", text="×", id_=f"remove-layer-{lid}")
        remove_btn.title = "Remove this layer"
        neuron_box.appendChild(remove_btn)

        layers_track_el.appendChild(neuron_box)

        select.addEventListener("change", create_proxy(
            lambda evt, lid=lid: on_activation_change(lid, evt.target.value)))
        remove_btn.addEventListener("click", create_proxy(lambda evt, lid=lid: on_remove_layer(lid)))
        help_btn.addEventListener("click", create_proxy(open_act_help))

    render_weight_badges()
    clear_grad_markers()
    render_loss_readout()


def on_activation_change(lid, value):
    network_model.set_layer_activation(lid, value)
    ui_refresh.on_topology_changed()


def on_remove_layer(lid):
    network_model.remove_layer(lid)
    ui_refresh.on_topology_changed()


# ── Live value rendering ──────────────────────────────────────────────────

def _source_symbol(idx):
    """The variable name feeding this layer: the raw input for the first
    layer, otherwise the previous layer's activation output."""
    return "x" if idx == 0 else f"a<sub>{idx}</sub>"


ARROW_SHAFT_WIDTH = 3.0
ARROW_HEAD_SIZE = 2.5
ARROW_SVG_WIDTH = 12.0


def _delta_arrow_html(delta, color):
    """A small direction/magnitude arrow next to a weight or bias -- an
    actual arrow (shaft + fixed-size head), not a glyph, so only its
    HEIGHT (shaft length) grows with the size of the change; its width
    never changes. Points up if the value just grew, down if it shrank."""
    if not delta:
        return ""
    shaft_len = 5.0 + 16.0 * math.tanh(abs(delta) * 3.0)
    total_h = shaft_len + ARROW_HEAD_SIZE + 3.0
    cx = ARROW_SVG_WIDTH / 2
    if delta > 0:
        tip_y, tail_y, dy = 2.0, 2.0 + shaft_len, -shaft_len
    else:
        tip_y, tail_y, dy = total_h - 2.0, total_h - 2.0 - shaft_len, shaft_len
    shaft_d = f"M{cx:.1f},{tail_y:.1f} L{cx:.1f},{tip_y:.1f}"
    head_d = _arrowhead_path_d(cx, tip_y, 0.0, dy, size=ARROW_HEAD_SIZE)
    path_d = f"{shaft_d} {head_d}" if head_d else shaft_d
    return (
        f"<svg class='delta-arrow-svg' width='{ARROW_SVG_WIDTH:.0f}' height='{total_h:.1f}' "
        f"viewBox='0 0 {ARROW_SVG_WIDTH:.0f} {total_h:.1f}' "
        f"title='changed by {delta:+.3f} last update'>"
        f"<path d='{path_d}' stroke='{color}' stroke-width='{ARROW_SHAFT_WIDTH:.1f}' "
        f"stroke-linecap='round' fill='none'/></svg>"
    )


def _weight_badge_html(layer, idx):
    """Renders the layer's weight (and bias, if enabled) as a single
    algebraic equation -- e.g. "5.00x" or "1.20a1 + 0.30" -- instead of a
    bare "w = 5" label."""
    source = _source_symbol(idx)
    w_arrow = _delta_arrow_html(layer.get("last_delta_w"), "var(--teal-dark)")
    html = f"<span class='w-color'>{fmt(layer['w'], 2)}</span>{source}{w_arrow}"
    if state.biases_enabled:
        b = layer["b"]
        sign = "+" if b >= 0 else "−"
        b_arrow = _delta_arrow_html(layer.get("last_delta_b"), "var(--orange-dark)")
        html += f" <span class='eq-op'>{sign}</span> <span class='b-color'>{fmt(abs(b), 2)}</span>{b_arrow}"
    return html


def render_weight_badges():
    for idx, layer in enumerate(state.layers):
        badge = state.get_id(f"weight-badge-{layer['id']}")
        if badge:
            badge.innerHTML = _weight_badge_html(layer, idx)


def render_loss_readout():
    if loss_readout_el is None:
        return
    if state.forward_cache is not None:
        loss_readout_el.textContent = fmt(state.forward_cache["mean_loss"])
    else:
        loss_readout_el.textContent = "–"


LOSS_PULSE_DURATION = 0.5


async def pulse_loss():
    box = state.get_id("loss-node-box")
    if box is None:
        return
    box.classList.add("loss-pulse")
    await asyncio.sleep(LOSS_PULSE_DURATION)
    box.classList.remove("loss-pulse")


def set_box_active(box_id, active):
    box = state.get_id(box_id)
    if box:
        if active:
            box.classList.add("active-highlight")
        else:
            box.classList.remove("active-highlight")


def clear_all_highlights():
    set_box_active("output-node-box", False)
    for layer in state.layers:
        set_box_active(f"neuron-box-{layer['id']}", False)
        set_box_active(f"weight-badge-{layer['id']}", False)


# ── Backward-reveal arrows + chain-rule labels ────────────────────────────
#
# Each revealed sub-step draws one curved arrow running from an edge of the
# box the gradient just arrived from to the matching edge of the box it's
# arriving at now, with its chain-rule formula centered under (or over) the
# dip of that arrow -- not floating arbitrarily. Consecutive markers
# alternate above/below the diagram so labels have room to breathe instead
# of stacking up on one side. Markers accumulate across an epoch's reveal
# and are cleared together at the next forward pass.
#
# The walk starts at the new Loss node (dL/dŷ, the boundary hop), then for
# each layer visits its ACTIVATION box before its LINEAR/weight box:
#     dL/da_i = dL/dn_{i+1} * dn_{i+1}/da_i   -- arriving at the activation node
#     dL/dn_i = dL/da_i * da_i/dn_i           -- arriving at the linear node
#     dL/dw_i = dL/dn_i * dn_i/dw_i           -- shown alongside dL/dn_i
# Note da_i/dn_i (this activation's own local slope) is revealed at the
# LINEAR node's arrival, not the activation node's -- it's the factor that
# turns the just-arrived dL/da_i into dL/dn_i, so it belongs to the hop
# that consumes dL/da_i, one box further back.

def marker_endpoints(plan_idx, is_activation):
    """(source_box_id, target_box_id) for a given plan index/stage,
    computed purely from plan order -- each arrow's source is always the
    box the previous arrow in the backward walk pointed at."""
    entry = state.plan[plan_idx]
    lid = entry["layer_id"]
    if is_activation:
        target_id = f"neuron-box-{lid}"
        if plan_idx == 0:
            source_id = "output-node-box"
        else:
            prev_lid = state.plan[plan_idx - 1]["layer_id"]
            source_id = f"weight-badge-{prev_lid}"
    else:
        source_id = f"neuron-box-{lid}"
        target_id = f"weight-badge-{lid}"
    return source_id, target_id


_marker_els = []  # index-aligned with state.grad_markers: {"path", "head", "label"} or None


def add_grad_marker(source_id, target_id, html):
    above = len(state.grad_markers) % 2 == 1
    state.grad_markers.append({
        "source_id": source_id, "target_id": target_id, "html": html, "above": above,
    })
    redraw_grad_markers()


def clear_grad_markers():
    state.grad_markers = []
    _marker_els.clear()
    if grad_arrow_svg_el is not None:
        grad_arrow_svg_el.innerHTML = ""
    if grad_label_layer_el is not None:
        grad_label_layer_el.innerHTML = ""


def _arrowhead_path_d(x2, y2, dx, dy, size=8.0):
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return None
    ux, uy = dx / mag, dy / mag
    spread = math.radians(28)
    cs, sn = math.cos(spread), math.sin(spread)
    wx1, wy1 = ux * cs - uy * sn, ux * sn + uy * cs
    wx2, wy2 = ux * cs + uy * sn, -ux * sn + uy * cs
    ax1, ay1 = x2 - size * wx1, y2 - size * wy1
    ax2, ay2 = x2 - size * wx2, y2 - size * wy2
    return f"M{x2:.1f},{y2:.1f} L{ax1:.1f},{ay1:.1f} M{x2:.1f},{y2:.1f} L{ax2:.1f},{ay2:.1f}"


def _svg_path(d, stroke_width=2.0):
    p = document.createElementNS(SVG_NS, "path")
    p.setAttribute("d", d)
    p.setAttribute("fill", "none")
    p.setAttribute("stroke", ARROW_COLOR)
    p.setAttribute("stroke-width", str(stroke_width))
    p.setAttribute("stroke-linecap", "round")
    return p


def redraw_grad_markers():
    """Recomputes every marker's arrow + label position from the LIVE
    positions of its source/target boxes. Safe to call any time (resize,
    new marker). Reuses each existing marker's DOM nodes rather than
    tearing down and rebuilding all of them -- only a brand-new marker
    gets fresh nodes (and so is the only one that plays the "row-in"
    entrance animation). Recreating every node on each call, as a full
    clear-and-redraw would, restarts that animation on markers that were
    already on screen, which is what made the whole set flash on every
    reveal instead of just the newest one appearing.

    Callers are responsible for having already applied any DOM change
    that would resize a box (e.g. re-rendering a weight badge's text)
    BEFORE calling this, so the measured position is the box's FINAL one
    -- measuring first and mutating content after is what causes a label
    to visibly jump right after it appears."""
    if grad_arrow_svg_el is None or grad_label_layer_el is None or diagram_canvas_el is None:
        return
    while len(_marker_els) < len(state.grad_markers):
        _marker_els.append(None)

    canvas_rect = diagram_canvas_el.getBoundingClientRect()

    for i, marker in enumerate(state.grad_markers):
        source_el = state.get_id(marker["source_id"])
        target_el = state.get_id(marker["target_id"])
        if source_el is None or target_el is None:
            continue

        above = marker.get("above", False)
        sr = source_el.getBoundingClientRect()
        tr = target_el.getBoundingClientRect()
        x1 = sr.left - canvas_rect.left + sr.width / 2
        x2 = tr.left - canvas_rect.left + tr.width / 2
        if above:
            y1 = sr.top - canvas_rect.top
            y2 = tr.top - canvas_rect.top
            dip = min(y1, y2) - 34
        else:
            y1 = sr.top - canvas_rect.top + sr.height
            y2 = tr.top - canvas_rect.top + tr.height
            dip = max(y1, y2) + 34

        path_d = f"M{x1:.1f},{y1:.1f} Q{(x1 + x2) / 2:.1f},{dip:.1f} {x2:.1f},{y2:.1f}"
        head_d = _arrowhead_path_d(x2, y2, x2 - (x1 + x2) / 2, y2 - dip)
        label_left = f"{(x1 + x2) / 2:.1f}px"
        label_top = f"{(dip - 6) if above else (dip + 6):.1f}px"

        els = _marker_els[i]
        if els is None:
            path_el = _svg_path(path_d)
            grad_arrow_svg_el.appendChild(path_el)
            head_el = _svg_path(head_d) if head_d else None
            if head_el is not None:
                grad_arrow_svg_el.appendChild(head_el)
            label = make_el("div", "grad-label above" if above else "grad-label")
            label.innerHTML = marker["html"]
            label.style.left = label_left
            label.style.top = label_top
            grad_label_layer_el.appendChild(label)
            _marker_els[i] = {"path": path_el, "head": head_el, "label": label}
        else:
            els["path"].setAttribute("d", path_d)
            if head_d:
                if els["head"] is None:
                    els["head"] = _svg_path(head_d)
                    grad_arrow_svg_el.appendChild(els["head"])
                else:
                    els["head"].setAttribute("d", head_d)
            elif els["head"] is not None:
                els["head"].remove()
                els["head"] = None
            els["label"].style.left = label_left
            els["label"].style.top = label_top


def render_boundary_reveal():
    """The very first backward reveal each epoch: the loss's own gradient
    arriving at ŷ. Not a chain-rule product -- there's nothing further
    upstream than the loss -- so it's shown as the direct derivative of
    the loss formula instead of a formula with two factors."""
    html = "<div class='grad-label-formula'>dL/dŷ = 2(ŷ − y)</div>"
    add_grad_marker("loss-node-box", "output-node-box", html)
    clear_all_highlights()
    set_box_active("output-node-box", True)


def render_activation_reveal(entry, plan_idx):
    """Reveals the activation node's half of the chain rule:
    dL/da_i = dL/dn_{i+1} * dn_{i+1}/da_i (or dL/dŷ * dŷ/da_i for the last
    layer, since ŷ IS a_L with no transform in between). Purely visual --
    never mutates a weight."""
    pos = entry["layer_pos"]
    if plan_idx == 0:
        formula = f"dL/da<sub>{pos}</sub> = dL/dŷ · dŷ/da<sub>{pos}</sub>"
    else:
        prev_pos = state.plan[plan_idx - 1]["layer_pos"]
        formula = (f"dL/da<sub>{pos}</sub> = dL/dn<sub>{prev_pos}</sub> · "
                   f"dn<sub>{prev_pos}</sub>/da<sub>{pos}</sub>")
    html = f"<div class='grad-label-formula'>{formula}</div>"

    source_id, target_id = marker_endpoints(plan_idx, is_activation=True)
    add_grad_marker(source_id, target_id, html)

    clear_all_highlights()
    set_box_active(target_id, True)


def render_linear_reveal(entry, plan_idx):
    """Reveals the linear/weight node's half of the chain rule:
    dL/dn_i = dL/da_i * da_i/dn_i, and dL/dw_i = dL/dn_i * dn_i/dw_i (the
    weight update itself) -- then pulses the weight badge. Call AFTER the
    caller has already applied the weight update to state.layers (this
    function only ever renders, it never mutates)."""
    pos = entry["layer_pos"]
    n_formula = f"dL/dn<sub>{pos}</sub> = dL/da<sub>{pos}</sub> · da<sub>{pos}</sub>/dn<sub>{pos}</sub>"
    w_formula = f"dL/dw<sub>{pos}</sub> = dL/dn<sub>{pos}</sub> · dn<sub>{pos}</sub>/dw<sub>{pos}</sub>"
    bias_line = ""
    if state.biases_enabled:
        bias_line = (
            f"<div class='grad-label-formula'>dL/db<sub>{pos}</sub> = dL/dn<sub>{pos}</sub> "
            f"· dn<sub>{pos}</sub>/db<sub>{pos}</sub></div>"
        )
    html = (f"<div class='grad-label-formula'>{n_formula}</div>"
            f"<div class='grad-label-formula'>{w_formula}</div>{bias_line}")

    # Update the badge's content BEFORE measuring/drawing -- if the badge's
    # size changes (a new digit, a delta-arrow appearing for the first
    # time) after the arrow/label were already positioned, the anchor
    # point moves out from under them and the label visibly jumps.
    render_weight_badges()

    source_id, target_id = marker_endpoints(plan_idx, is_activation=False)
    add_grad_marker(source_id, target_id, html)

    clear_all_highlights()
    set_box_active(target_id, True)
    asyncio.ensure_future(pulse_weight(entry["layer_id"]))


PULSE_DURATION = 0.45


async def pulse_weight(lid):
    badge = state.get_id(f"weight-badge-{lid}")
    if badge is None:
        return
    badge.classList.add("weight-pulse")
    await asyncio.sleep(PULSE_DURATION)
    badge.classList.remove("weight-pulse")


# ── Activation help popover ───────────────────────────────────────────────

def open_act_help(evt=None):
    popover = state.get_id("act-help-popover")
    if popover:
        popover.classList.remove("hidden")


def close_act_help(evt=None):
    popover = state.get_id("act-help-popover")
    if popover:
        popover.classList.add("hidden")
