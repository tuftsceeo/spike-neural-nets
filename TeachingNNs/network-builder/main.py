"""
Neural Network Builder — main.py
PyScript 2026.3.1
"""

import math
from pyscript import document, window, when
from pyscript.ffi import create_proxy
from Device import Element

# ── State ──────────────────────────────────────────────────────────────────────

rows: list[dict] = []
row_counter      = 0
devices: list[str] = []
is_running       = False

ACTIVATION_OPTIONS = [
    ("None",     ""),
    ("ReLU",     "relu"),
    ("Sigmoid",  "sigmoid"),
    ("Tanh",     "tanh"),
    ("Linear",   "linear"),
    ("Step",     "step"),
    ("Softplus", "softplus"),
]

CHANNELS = [
    ("— channel —", ""),
    ("Channel 1",   "ch1"),
    ("Channel 2",   "ch2"),
    ("Channel 3",   "ch3"),
    ("Channel 4",   "ch4"),
]

ARROW_COLOR = "#1e40af"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_id(id_: str):
    return document.getElementById(id_)

def row_by_id(rid: int) -> dict | None:
    for r in rows:
        if r["id"] == rid:
            return r
    return None

def device_options_html() -> str:
    opts = '<option value="">— device —</option>'
    for name in devices:
        opts += f'<option value="{name}">{name}</option>'
    return opts

def channel_options_html() -> str:
    return "".join(
        f'<option value="{v}">{label}</option>' for label, v in CHANNELS
    )

def populate_act_select():
    sel = get_id("act-select")
    if not sel:
        return
    html = ""
    for label, val in ACTIVATION_OPTIONS:
        html += f'<option value="{val}">{label}</option>'
    sel.innerHTML = html

# ── SVG arrow primitives ───────────────────────────────────────────────────────

def svg_path(d: str, stroke_w: float = 2.0):
    p = document.createElementNS("http://www.w3.org/2000/svg", "path")
    p.setAttribute("d", d)
    p.setAttribute("fill", "none")
    p.setAttribute("stroke", ARROW_COLOR)
    p.setAttribute("stroke-width", str(stroke_w))
    p.setAttribute("stroke-linecap", "round")
    p.setAttribute("stroke-linejoin", "round")
    return p

def arrowhead(svg_el, x2: float, y2: float, dx: float, dy: float,
              stroke_w: float = 2.0, size: float = 9.0):
    """V-shaped arrowhead at (x2,y2) pointing in direction (dx,dy)."""
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-9:
        return
    ux, uy = dx / mag, dy / mag
    spread = math.radians(26)
    cs, sn = math.cos(spread), math.sin(spread)
    wx1, wy1 =  ux * cs - uy * sn,  ux * sn + uy * cs
    wx2, wy2 =  ux * cs + uy * sn, -ux * sn + uy * cs
    ax1, ay1 = x2 - size * wx1, y2 - size * wy1
    ax2, ay2 = x2 - size * wx2, y2 - size * wy2
    d = f"M{x2:.2f},{y2:.2f} L{ax1:.2f},{ay1:.2f} M{x2:.2f},{y2:.2f} L{ax2:.2f},{ay2:.2f}"
    svg_el.appendChild(svg_path(d, stroke_w))

def straight_arrow(svg_el, x1, y1, x2, y2, stroke_w: float = 2.0):
    """Straight line with arrowhead."""
    d = f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}"
    svg_el.appendChild(svg_path(d, stroke_w))
    arrowhead(svg_el, x2, y2, x2 - x1, y2 - y1, stroke_w)

# ── Main redraw ────────────────────────────────────────────────────────────────

def redraw_arrows():
    svg_el  = get_id("arrow-svg")
    wrap_el = get_id("network-wrap")
    if not svg_el or not wrap_el or not rows:
        if svg_el:
            svg_el.innerHTML = ""
        return

    svg_el.innerHTML = ""
    wr = wrap_el.getBoundingClientRect()

    def rel(el):
        r = el.getBoundingClientRect()
        return {
            "x": r.left - wr.left,
            "y": r.top  - wr.top,
            "w": r.width,
            "h": r.height,
        }

    in_pts   = []
    eq_pts   = []
    eq_r_pts = []
    out_pts  = []

    for row in rows:
        rid    = row["id"]
        in_el  = get_id(f"cell-in-{rid}")
        eq_el  = get_id(f"cell-eq-{rid}")
        out_el = get_id(f"cell-out-{rid}")
        if in_el and eq_el and out_el:
            ir  = rel(in_el)
            er  = rel(eq_el)
            or_ = rel(out_el)
            in_pts.append(  (ir["x"] + ir["w"],  ir["y"] + ir["h"] / 2, rid) )
            eq_pts.append(  (er["x"],             er["y"] + er["h"] / 2, rid) )
            eq_r_pts.append((er["x"] + er["w"],   er["y"] + er["h"] / 2, rid) )
            out_pts.append( (or_["x"],             or_["y"] + or_["h"] / 2, rid) )

    if not in_pts:
        return

    act_el = get_id("act-box")

    # ── 1. Input → Equation: straight lines, spread at both ends ─────────────
    SPREAD_FRAC = 0.55
    n_rows = len(in_pts)

    in_heights = {}
    eq_heights = {}
    for row in rows:
        rid   = row["id"]
        in_el = get_id(f"cell-in-{rid}")
        eq_el = get_id(f"cell-eq-{rid}")
        if in_el:
            in_heights[rid] = rel(in_el)["h"]
        if eq_el:
            eq_heights[rid] = rel(eq_el)["h"]

    for src_idx, (ix, iy, i_rid) in enumerate(in_pts):
        for tgt_idx, (ex, ey, e_rid) in enumerate(eq_pts):
            if n_rows > 1:
                src_h    = in_heights.get(i_rid, 120)
                spread_h = min(src_h * SPREAD_FRAC, (n_rows - 1) * 14)
                tail_y   = iy - spread_h / 2 + (tgt_idx / (n_rows - 1)) * spread_h

                eq_h     = eq_heights.get(e_rid, 80)
                spread_h = min(eq_h * SPREAD_FRAC, (n_rows - 1) * 14)
                head_y   = ey - spread_h / 2 + (src_idx / (n_rows - 1)) * spread_h
            else:
                tail_y = iy
                head_y = ey

            same = i_rid == e_rid
            straight_arrow(svg_el, ix, tail_y, ex, head_y,
                           stroke_w=2.2 if same else 1.8)

    # ── 2. Equation → Activation: horizontal straight arrows ─────────────────
    if act_el and eq_r_pts:
        act_x = rel(act_el)["x"]
        for (ex, ey, e_rid) in eq_r_pts:
            straight_arrow(svg_el, ex, ey, act_x, ey, stroke_w=2.0)

    # ── 3. Activation → Output: horizontal straight arrows ───────────────────
    if act_el and out_pts:
        act_rx = rel(act_el)["x"] + rel(act_el)["w"]
        for (ox, oy, o_rid) in out_pts:
            straight_arrow(svg_el, act_rx, oy, ox, oy, stroke_w=2.0)

# ── Row HTML ───────────────────────────────────────────────────────────────────

def make_left_row_html(row: dict) -> str:
    rid  = row["id"]
    name = row["name"]

    eq_parts = ""
    for i, r in enumerate(rows):
        coeff_val = row["coeffs"][i] if i < len(row["coeffs"]) else 1.0
        if i > 0:
            eq_parts += '<span class="eq-op">+</span>'
        eq_parts += (
            f'<input type="number" step="any" value="{coeff_val:.2f}"'
            f' class="eq-num-input" id="coeff-{rid}-{i}"'
            f' data-row="{rid}" data-idx="{i}" />'
            f'<span class="eq-var" id="var-label-{rid}-{i}">{r["name"]}</span>'
        )
    eq_parts += (
        f'<span class="eq-op">+</span>'
        f'<input type="number" step="any" value="{row["bias"]:.2f}"'
        f' class="eq-bias-input" id="bias-{rid}" data-row="{rid}" />'
    )

    dev_opts = device_options_html()
    ch_opts  = channel_options_html()

    return f"""
<div class="neuron-row neuron-row-left" id="row-left-{rid}" data-row="{rid}">

    <div class="cell-label">
        <input type="text" class="name-input" id="name-{rid}"
               value="{name}" data-row="{rid}" maxlength="12" />
    </div>

    <div class="cell-node" id="cell-in-{rid}">
        <div class="node-card input-node">
            <div class="node-header">
                <select class="node-device-select" id="dev-in-{rid}">{dev_opts}</select>
                <span class="node-reading" id="reading-in-{rid}">—</span>
            </div>
            <div class="node-plot">
                <canvas class="plot-canvas" id="canvas-in-{rid}" width="170" height="70"></canvas>
            </div>
            <div class="plot-footer">
                <select class="channel-select" id="chan-in-{rid}">{ch_opts}</select>
            </div>
        </div>
    </div>

    <div class="cell-arrow-gap-in"></div>

    <div class="cell-node" id="cell-eq-{rid}">
        <div class="eq-node" id="eq-node-{rid}">
            <div class="eq-inline" id="eq-inline-{rid}">
                {eq_parts}
            </div>
        </div>
    </div>

</div>
"""

def make_right_row_html(row: dict) -> str:
    rid      = row["id"]
    dev_opts = device_options_html()
    ch_opts  = channel_options_html()

    return f"""
<div class="neuron-row neuron-row-right" id="row-right-{rid}" data-row="{rid}">

    <div class="cell-node" id="cell-out-{rid}">
        <div class="node-card output-node">
            <div class="node-header">
                <select class="node-device-select" id="dev-out-{rid}">{dev_opts}</select>
                <span class="node-reading" id="reading-out-{rid}">—</span>
            </div>
            <div class="node-plot">
                <canvas class="plot-canvas" id="canvas-out-{rid}" width="170" height="70"></canvas>
            </div>
            <div class="plot-footer">
                <select class="channel-select" id="chan-out-{rid}">{ch_opts}</select>
            </div>
        </div>
    </div>

    <div class="cell-delete">
        <button class="btn-delete-row" id="del-{rid}" data-row="{rid}" title="Remove neuron">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5">
                <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>
            </svg>
        </button>
    </div>

</div>
"""

# ── Equation field sync ────────────────────────────────────────────────────────

def sync_all_eq_fields():
    for row in rows:
        rid       = row["id"]
        container = get_id(f"eq-inline-{rid}")
        if not container:
            continue
        while len(row["coeffs"]) < len(rows):
            row["coeffs"].append(1.0)
        html = ""
        for i, r in enumerate(rows):
            coeff_val = row["coeffs"][i]
            if i > 0:
                html += '<span class="eq-op">+</span>'
            html += (
                f'<input type="number" step="any" value="{coeff_val:.2f}"'
                f' class="eq-num-input" id="coeff-{rid}-{i}"'
                f' data-row="{rid}" data-idx="{i}" />'
                f'<span class="eq-var" id="var-label-{rid}-{i}">{r["name"]}</span>'
            )
        html += (
            f'<span class="eq-op">+</span>'
            f'<input type="number" step="any" value="{row["bias"]:.2f}"'
            f' class="eq-bias-input" id="bias-{rid}" data-row="{rid}" />'
        )
        container.innerHTML = html
        bind_eq_inputs(rid)

def sync_var_labels():
    for i, r in enumerate(rows):
        label = r["name"]
        for row in rows:
            el = get_id(f"var-label-{row['id']}-{i}")
            if el:
                el.textContent = label

# ── Event binding ──────────────────────────────────────────────────────────────

def bind_eq_inputs(rid: int):
    row = row_by_id(rid)
    if not row:
        return
    for i in range(len(rows)):
        inp = get_id(f"coeff-{rid}-{i}")
        if inp:
            def make_ch(r, idx):
                def h(evt):
                    try:
                        r["coeffs"][idx] = float(evt.target.value)
                    except (ValueError, TypeError):
                        pass
                return create_proxy(h)
            inp.addEventListener("input", make_ch(row, i))
    bias_inp = get_id(f"bias-{rid}")
    if bias_inp:
        def make_bh(r):
            def h(evt):
                try:
                    r["bias"] = float(evt.target.value)
                except (ValueError, TypeError):
                    pass
            return create_proxy(h)
        bias_inp.addEventListener("input", make_bh(row))

def bind_row_events(rid: int):
    row = row_by_id(rid)
    if not row:
        return

    name_el = get_id(f"name-{rid}")
    if name_el:
        def make_nh(r):
            def h(evt):
                new = evt.target.value.strip() or r["name"]
                r["name"] = new
                sync_var_labels()
            return create_proxy(h)
        name_el.addEventListener("input", make_nh(row))

    del_btn = get_id(f"del-{rid}")
    if del_btn:
        def make_dh(r):
            def h(evt):
                delete_row(r["id"])
            return create_proxy(h)
        del_btn.addEventListener("click", make_dh(row))

    bind_eq_inputs(rid)

# ── Row CRUD ───────────────────────────────────────────────────────────────────

def add_row(evt=None):
    global row_counter
    row_counter += 1
    rid = row_counter
    n   = len(rows) + 1
    row = {
        "id":          rid,
        "name":        f"x{n}",
        "coeffs":      [1.0] * n,
        "bias":        0.0,
        "input_data":  [],
        "output_data": [],
    }
    rows.append(row)

    left_container = get_id("rows-container")
    lw = document.createElement("div")
    lw.innerHTML = make_left_row_html(row)
    left_container.appendChild(lw.firstElementChild)

    right_container = get_id("rows-out-container")
    rw = document.createElement("div")
    rw.innerHTML = make_right_row_html(row)
    right_container.appendChild(rw.firstElementChild)

    bind_row_events(rid)
    sync_all_eq_fields()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)


def delete_row(rid: int):
    global rows

    for suffix in ("left", "right"):
        el = get_id(f"row-{suffix}-{rid}")
        if el:
            el.remove()

    idx = next((i for i, r in enumerate(rows) if r["id"] == rid), None)
    if idx is None:
        return

    rows = [r for r in rows if r["id"] != rid]

    for r in rows:
        if idx < len(r["coeffs"]):
            r["coeffs"].pop(idx)

    sync_all_eq_fields()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

# ── Device management ──────────────────────────────────────────────────────────

def add_device_chip(name: str):
    """Insert a chip for `name` above the connect button."""
    dl  = get_id("device-list")
    btn = get_id("add-device-btn")

    chip = document.createElement("div")
    chip.className = "device-row"
    chip.id = f"chip-{name}"
    chip.innerHTML = (
        f'<div class="device-indicator"></div>'
        f'<span class="device-name">{name}</span>'
        f'<button class="btn-disconnect" title="Disconnect">'
        f'    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"'
        f'         stroke="currentColor" stroke-width="2.5">'
        f'        <path d="M18 6L6 18M6 6l12 12"/>'
        f'    </svg>'
        f'</button>'
    )

    disc = chip.querySelector(".btn-disconnect")

    def make_disc(dev_name):
        def handler(evt):
            devices.remove(dev_name)
            chip_el = get_id(f"chip-{dev_name}")
            if chip_el:
                chip_el.remove()
            refresh_device_dropdowns()
        return create_proxy(handler)

    disc.addEventListener("click", make_disc(name))

    # Insert before the button so chips stack above it
    dl.insertBefore(chip, btn)
    refresh_device_dropdowns()


def refresh_device_dropdowns():
    dev_opts = device_options_html()
    for row in rows:
        rid = row["id"]
        for pfx in ("dev-in-", "dev-out-"):
            sel = get_id(f"{pfx}{rid}")
            if sel:
                cur = sel.value
                sel.innerHTML = dev_opts
                sel.value = cur


async def create_new_device(evt=None):
    test_dev = Element()
    await test_dev.connect()
    name = test_dev.name if hasattr(test_dev, "name") else f"Device {len(devices) + 1}"
    devices.append(name)
    add_device_chip(name)

# ── Play / Stop ────────────────────────────────────────────────────────────────

def play_network(evt=None):
    global is_running
    is_running = True
    get_id("play-btn").setAttribute("disabled", "")
    get_id("stop-btn").removeAttribute("disabled")
    document.body.classList.add("running")

def stop_network(evt=None):
    global is_running
    is_running = False
    get_id("stop-btn").setAttribute("disabled", "")
    get_id("play-btn").removeAttribute("disabled")
    document.body.classList.remove("running")

# ── Activation help popover ────────────────────────────────────────────────────

def open_act_help(evt=None):
    get_id("act-help-popover").classList.remove("hidden")

def close_act_help(evt=None):
    get_id("act-help-popover").classList.add("hidden")

# ── Resize observer ────────────────────────────────────────────────────────────

def setup_resize_observer():
    wrap = get_id("network-wrap")
    if not wrap:
        return
    def on_resize(entries, observer):
        redraw_arrows()
    observer = window.ResizeObserver.new(create_proxy(on_resize))
    observer.observe(wrap)

# ── Static wiring ──────────────────────────────────────────────────────────────

@when("click", "#add-device-btn")
async def _on_add_device(evt):
    await create_new_device()

@when("click", "#add-row-btn")
def _on_add_row(evt):
    add_row()

@when("click", "#play-btn")
def _on_play(evt):
    play_network()

@when("click", "#stop-btn")
def _on_stop(evt):
    stop_network()

@when("click", "#act-help-btn")
def _on_act_help(evt):
    open_act_help()

@when("click", "#close-act-help-btn")
def _on_close_act_help(evt):
    close_act_help()

# ── Boot ───────────────────────────────────────────────────────────────────────

def boot():
    get_id("loading-splash").style.display = "none"
    get_id("page-wrap").style.display = "flex"

    populate_act_select()
    add_row()

    setup_resize_observer()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 120)

boot()

# ── WASM Worker patch ──────────────────────────────────────────────────────────

import asyncio
import legoeducation.background_worker as bw

def wasm_start_thread(self):
    self.loop = asyncio.get_event_loop()
    self.loop_ready.set()
    asyncio.ensure_future(_wasm_worker_loop(self))

def wasm_put_request(self, request):
    asyncio.ensure_future(self.async_put_request(request))

bw.Worker.start_thread = wasm_start_thread
bw.Worker.put_request  = wasm_put_request

async def _wasm_worker_loop(worker):
    worker._myble_registry = {}

    while True:
        try:
            req = await worker.request_queue.get()
            if req is None:
                break

            topic = req.get("topic")

            if topic == "send":
                device  = req.get("msg")
                message = req.get("msg2")
                myble   = worker._myble_registry.get(id(device))
                if myble is not None and message is not None:
                    try:
                        await myble.send(list(message))
                    except Exception as e:
                        print(f"BLE send error: {e}")

            elif topic == "connect":
                connect_callback = req.get("msg3")
                if connect_callback:
                    connect_callback(True)

            elif topic == "disconnect":
                device = req.get("msg")
                myble  = worker._myble_registry.pop(id(device), None)
                if myble is not None:
                    myble.disconnect()

            elif topic == "scan":
                callback = req.get("msg2")
                if callback:
                    callback([])

        except Exception as e:
            print(f"Worker loop error: {e}")