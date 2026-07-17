"""Training-data table rendering/binding. Data mutation itself
(add_data_point/remove_data_point) lives in network_model.py -- this file
only renders/binds the table and re-renders after each mutation."""
from pyscript.ffi import create_proxy

import state
from state import get_id
import network_model
import templates

def render_dataset_header():
    container = get_id("dataset-header")
    if not container:
        return
    html = ""
    for inp in state.inputs:
        html += f'<span class="dataset-row-header-x">{inp["name"]}</span>'
    for idx, out in enumerate(state.outputs):
        html += f'<span class="dataset-row-header-y">y{idx + 1}</span>'
    html += "<span></span>"
    container.innerHTML = html

def render_dataset_table():
    container = get_id("dataset-rows")
    if not container:
        return
    html = ""
    for p in state.training_data:
        pid = p["id"]
        html += f'<div class="dataset-row" data-id="{pid}">'
        for inp in state.inputs:
            iid = inp["id"]
            val = p["xs"].get(iid, 0.0)
            html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-x" '
                      f'id="point-x-{pid}-{iid}" value="{val}" />')
        for out in state.outputs:
            oid = out["id"]
            val = p["ys"].get(oid, 0.0)
            html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-y" '
                      f'id="point-y-{pid}-{oid}" value="{val}" />')
        html += f'<button class="btn-remove-point" id="del-point-{pid}" title="Remove point">{templates.delete_x_svg()}</button>'
        html += "</div>"
    container.innerHTML = html
    for p in state.training_data:
        bind_dataset_row_events(p["id"])

def bind_dataset_row_events(pid: int):
    for inp in state.inputs:
        iid = inp["id"]
        x_el = get_id(f"point-x-{pid}-{iid}")
        if x_el:
            def hx(evt, pid=pid, iid=iid):
                p = next((pt for pt in state.training_data if pt["id"] == pid), None)
                try:
                    if p:
                        p["xs"][iid] = float(evt.target.value)
                        refresh_dataset_plot_points()
                except (ValueError, TypeError):
                    pass
            x_el.addEventListener("input", create_proxy(hx))

    for out in state.outputs:
        oid = out["id"]
        y_el = get_id(f"point-y-{pid}-{oid}")
        if y_el:
            def hy(evt, pid=pid, oid=oid):
                p = next((pt for pt in state.training_data if pt["id"] == pid), None)
                try:
                    if p:
                        p["ys"][oid] = float(evt.target.value)
                        refresh_dataset_plot_points()
                except (ValueError, TypeError):
                    pass
            y_el.addEventListener("input", create_proxy(hy))

    del_btn = get_id(f"del-point-{pid}")
    if del_btn:
        del_btn.addEventListener("click", create_proxy(lambda evt, pid=pid: _on_remove_point(pid)))

def _on_remove_point(pid: int):
    network_model.remove_data_point(pid)
    render_dataset_table()
    refresh_dataset_plot_points()

def refresh_dataset_plot_points():
    """Push each output's Data trace: y = that output's column, x = the
    FIRST input's column (the fit graph's x-axis is fixed to it)."""
    fit_plot_obj = state.all_plots.get("plot-fit")
    if not fit_plot_obj or not state.inputs:
        return
    first_iid = state.inputs[0]["id"]
    xs = [p["xs"].get(first_iid, 0.0) for p in state.training_data]
    for out in state.outputs:
        oid = out["id"]
        ys = [p["ys"].get(oid, 0.0) for p in state.training_data]
        fit_plot_obj.update_data_points(oid, xs, ys)

def render_add_point_row():
    container = get_id("dataset-add-row")
    if not container:
        return
    html = ""
    for inp in state.inputs:
        iid = inp["id"]
        html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-x" '
                  f'id="new-point-x-{iid}" placeholder="{inp["name"]}" />')
    for idx, out in enumerate(state.outputs):
        oid = out["id"]
        html += (f'<input type="number" step="any" class="dataset-num-input dataset-num-input-y" '
                  f'id="new-point-y-{oid}" placeholder="y{idx + 1}" />')
    html += ('<button class="btn-add-point" id="add-point-btn" title="Add this point to the dataset">'
             '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="2.5"><path d="M12 5v14M5 12h14"/></svg>'
             'Add point</button>')
    container.innerHTML = html
    btn = get_id("add-point-btn")
    if btn:
        btn.addEventListener("click", create_proxy(on_add_point_click))

def on_add_point_click(evt=None):
    xs = {}
    for inp in state.inputs:
        iid = inp["id"]
        el = get_id(f"new-point-x-{iid}")
        try:
            xs[iid] = float(el.value) if el and el.value.strip() != "" else None
        except ValueError:
            xs[iid] = None

    ys = {}
    for out in state.outputs:
        oid = out["id"]
        el = get_id(f"new-point-y-{oid}")
        try:
            ys[oid] = float(el.value) if el and el.value.strip() != "" else None
        except ValueError:
            ys[oid] = None

    if any(v is None for v in xs.values()) or any(v is None for v in ys.values()):
        print("Enter every x and y value before adding a point.")
        return

    network_model.add_data_point(xs, ys)
    render_dataset_table()
    refresh_dataset_plot_points()

    for inp in state.inputs:
        el = get_id(f"new-point-x-{inp['id']}")
        if el:
            el.value = ""
    for out in state.outputs:
        el = get_id(f"new-point-y-{out['id']}")
        if el:
            el.value = ""
