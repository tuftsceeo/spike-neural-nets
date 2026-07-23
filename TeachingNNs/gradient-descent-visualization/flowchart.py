from js import document

from state import state
from dom import el, point_pass_col_el, flowchart_panel_el, live_network_eq_el
from math_core import fmt, ARROW_GLYPH, error_label
from plots import render_slice_plots

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


def build_point_pass_skeleton():
    """Builds the small per-point forward-pass diagrams (p1, p2), stacked
    vertically and centered inside #point-pass-stack, below the network
    diagram. Starts with a plain-text "Input" / "Output" header row - no
    node-box padding/border, so its height matches the "Dataset" title
    exactly and the two columns stay tightly stacked in lockstep."""
    point_pass_col_el.innerHTML = ""

    header = make_el("div", "point-pass-header")
    header.appendChild(make_el("div", "point-io-label", text="Input"))
    header.appendChild(make_el("div", "flow-arrow point-io-spacer", text="→"))
    header.appendChild(make_el("div", "point-io-mid-spacer"))
    header.appendChild(make_el("div", "eq-op point-io-spacer", text="→"))
    header.appendChild(make_el("div", "point-io-label", text="Output"))
    point_pass_col_el.appendChild(header)

    for label in ("p1", "p2"):
        box = make_el("div", "point-calc-box state-inactive")
        box.id = f"point-box-{label}"

        diagram = make_el("div", "net-diagram")
        x_node = make_el("div", "node-card input-node", text="x = –")
        x_node.id = f"point-x-{label}"
        set_tip(x_node, "x — this point's input value")
        diagram.appendChild(x_node)
        diagram.appendChild(make_el("div", "flow-arrow", text="→"))
        lin = make_el("div", "eq-node linear-node", text="w·x + b")
        lin.id = f"point-lin-{label}"
        set_tip(lin, "ŷ = w·x + b — forward pass")
        diagram.appendChild(lin)
        diagram.appendChild(make_el("div", "eq-op", text="→"))
        out = make_el("div", "node-card output-node")
        out.id = f"point-out-{label}"
        out.innerHTML = "pred = – <span class='target-note'>(target –)</span>"
        set_tip(out, "ŷ — the model's prediction here")
        diagram.appendChild(out)

        box.appendChild(diagram)
        point_pass_col_el.appendChild(box)


def build_flow_skeleton():
    flowchart_panel_el.innerHTML = ""

    block = make_el("div", "epoch-block")
    block.id = "epoch-block"

    flow_wrap = make_el("div", "flowchart-wrap")
    flow_wrap.id = "flowchart-wrap"

    error_row = make_el("div", "flow-row flow-row-error state-inactive")
    error_row.id = "flow-row-error"
    error_node = make_el("div", "eq-node error-node", text="E = –")
    error_node.id = "error-node"
    set_tip(error_node, "E — average error across both points")
    error_row.appendChild(error_node)
    flow_wrap.appendChild(error_row)

    grad_row = make_el("div", "flow-row flow-row-grads state-inactive")
    grad_row.id = "flow-row-grads"
    dw = make_el("div", "eq-node grad-node w-color", text="dE/dw = –")
    dw.id = "grad-w-node"
    set_tip(dw, "w's gradient — how error responds to w")
    db = make_el("div", "eq-node grad-node b-color", text="dE/db = –")
    db.id = "grad-b-node"
    set_tip(db, "b's gradient — how error responds to b")
    grad_row.appendChild(dw)
    grad_row.appendChild(db)
    flow_wrap.appendChild(grad_row)

    update_row = make_el("div", "flow-row flow-row-updates state-inactive")
    update_row.id = "flow-row-updates"

    cell_w = make_el("div", "update-cell")
    node_w = make_el("div", "eq-node update-node w-color", text="Δw = –")
    node_w.id = "update-w-node"
    set_tip(node_w, "Δw = −learning rate × w's gradient")
    cell_w.appendChild(node_w)

    cell_b = make_el("div", "update-cell")
    node_b = make_el("div", "eq-node update-node b-color", text="Δb = –")
    node_b.id = "update-b-node"
    set_tip(node_b, "Δb = −learning rate × b's gradient")
    cell_b.appendChild(node_b)

    update_row.appendChild(cell_w)
    update_row.appendChild(cell_b)
    flow_wrap.appendChild(update_row)

    final_row = make_el("div", "flow-row flow-row-final state-inactive")
    final_row.id = "flow-row-final"
    final_w = make_el("div", "eq-node final-node w-color")
    final_w.id = "final-w-node"
    final_w.innerHTML = "<span class='dir-arrow'>–</span> w = –"
    set_tip(final_w, "new w = old w + Δw")
    final_b = make_el("div", "eq-node final-node b-color")
    final_b.id = "final-b-node"
    final_b.innerHTML = "<span class='dir-arrow'>–</span> b = –"
    set_tip(final_b, "new b = old b + Δb")
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
            f"<span class='w-color'>{fmt(w)}</span>·x + <span class='b-color'>{fmt(b)}</span>"
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
    el("update-w-node").textContent = f"Δw = {fmt(upd['delta_w'])}"
    el("update-b-node").textContent = f"Δb = {fmt(upd['delta_b'])}"


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
        f"{w_arrow}<span class='w-color'>{fmt(w)}</span>·x + "
        f"{b_arrow}<span class='b-color'>{fmt(b)}</span>"
    )
    if highlight:
        live_network_eq_el.classList.add("highlight-pulse")
    else:
        live_network_eq_el.classList.remove("highlight-pulse")


def reset_flow_display():
    """Blank the flowchart back to placeholders (used for the very
    first, pre-training snapshot)."""
    lin_html = f"<span class='w-color'>{fmt(state.w)}</span>·x + <span class='b-color'>{fmt(state.b)}</span>"
    el("point-x-p1").textContent = "x = –"
    el("point-x-p2").textContent = "x = –"
    el("point-lin-p1").innerHTML = lin_html
    el("point-lin-p2").innerHTML = lin_html
    el("point-out-p1").innerHTML = "pred = – <span class='target-note'>(target –)</span>"
    el("point-out-p2").innerHTML = "pred = – <span class='target-note'>(target –)</span>"
    el("error-node").textContent = "E = –"
    el("grad-w-node").textContent = "dE/dw = –"
    el("grad-b-node").textContent = "dE/db = –"
    el("update-w-node").textContent = "Δw = –"
    el("update-b-node").textContent = "Δb = –"
    el("final-w-node").innerHTML = "<span class='dir-arrow'>–</span> w = –"
    el("final-b-node").innerHTML = "<span class='dir-arrow'>–</span> b = –"
    set_all_states("inactive", "inactive", "inactive", "inactive", "inactive")
    render_slice_plots()
