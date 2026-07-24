"""Training-data table rendering/binding. Data mutation itself lives in
network_model.py -- this file only renders/binds the table and re-renders
after each mutation."""
from pyscript import document
from pyscript.ffi import create_proxy

import state
import network_model
import ui_refresh

rows_el = state.get_id("dataset-rows")
add_row_el = state.get_id("dataset-add-row")


def render_table():
    rows_el.innerHTML = ""
    for p in state.dataset:
        pid = p["id"]
        row = document.createElement("div")
        row.className = "dataset-row"

        x_input = document.createElement("input")
        x_input.type = "number"
        x_input.step = "any"
        x_input.className = "dataset-num-input dataset-num-input-x"
        x_input.value = str(p["x"])
        x_input.addEventListener("input", create_proxy(lambda evt, pid=pid: _on_edit(pid, "x", evt.target.value)))
        row.appendChild(x_input)

        y_input = document.createElement("input")
        y_input.type = "number"
        y_input.step = "any"
        y_input.className = "dataset-num-input dataset-num-input-y"
        y_input.value = str(p["y"])
        y_input.addEventListener("input", create_proxy(lambda evt, pid=pid: _on_edit(pid, "y", evt.target.value)))
        row.appendChild(y_input)

        del_btn = document.createElement("button")
        del_btn.className = "btn-remove-point"
        del_btn.textContent = "×"
        del_btn.title = "Remove point"
        del_btn.addEventListener("click", create_proxy(lambda evt, pid=pid: _on_remove(pid)))
        row.appendChild(del_btn)

        rows_el.appendChild(row)


def _on_edit(pid, field, raw_value):
    try:
        value = float(raw_value)
    except ValueError:
        return
    p = next((pt for pt in state.dataset if pt["id"] == pid), None)
    if p is None:
        return
    p[field] = value
    network_model.reset_training()
    ui_refresh.on_dataset_changed()


def _on_remove(pid):
    network_model.remove_data_point(pid)
    render_table()
    ui_refresh.on_dataset_changed()


def render_add_row():
    add_row_el.innerHTML = ""

    x_input = document.createElement("input")
    x_input.type = "number"
    x_input.step = "any"
    x_input.className = "dataset-num-input dataset-num-input-x"
    x_input.id = "new-point-x"
    x_input.placeholder = "x"
    add_row_el.appendChild(x_input)

    y_input = document.createElement("input")
    y_input.type = "number"
    y_input.step = "any"
    y_input.className = "dataset-num-input dataset-num-input-y"
    y_input.id = "new-point-y"
    y_input.placeholder = "y"
    add_row_el.appendChild(y_input)

    add_btn = document.createElement("button")
    add_btn.className = "btn-add-point"
    add_btn.id = "add-point-btn"
    add_btn.textContent = "+ Add point"
    add_btn.addEventListener("click", create_proxy(on_add_point_click))
    add_row_el.appendChild(add_btn)


def on_add_point_click(evt=None):
    x_el = state.get_id("new-point-x")
    y_el = state.get_id("new-point-y")
    try:
        x_val = float(x_el.value)
        y_val = float(y_el.value)
    except (ValueError, AttributeError):
        return

    network_model.add_data_point(x_val, y_val)
    render_table()
    x_el.value = ""
    y_el.value = ""
    ui_refresh.on_dataset_changed()
