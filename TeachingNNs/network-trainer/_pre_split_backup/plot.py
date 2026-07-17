from pyscript import document, window, when, Event
from pyscript.js_modules import Plotly
from collections import deque
import js
import json

class plot():
    def __init__(self, id):
        self.size = 100
        self.num = 12
        self.id = id
        self.buffer = []
        self.traces = js.JSON.parse(json.dumps([
            {'y': 0, 'mode': 'lines', 'name': 'Element Data', 'line': {'color': '#82ca9d'}}
        ]))
        for n in range(self.num):
            self.buffer.append(deque([], self.size))
        layout = js.JSON.parse(json.dumps({
            "margin": {"l": 20, "r": 10, "t": 10, "b": 20},
            "xaxis": {
                "title": {"font": {"size": 10}},
                "tickfont": {"size": 9},
                "showticklabels": False
            },
            "yaxis": {
                "tickfont": {"size": 9},
                "autorange": True
            },
            "showlegend": False
        }))
        Plotly.newPlot(self.id, self.traces, layout)

    def addPoints(self, num, data):
        lines = []
        num = (num - 1) * 2
        for i, d in enumerate(data):
            self.buffer[num + i].append(d)
            lines.append(list(self.buffer[num + i]))
        return js.JSON.parse(json.dumps({'y': lines}))

    def updatePlot(self, update):
        Plotly.update(self.id, update)


DATA_TRACE_COLORS = ['#82ca9d', '#22c55e', '#0891b2', '#84cc16', '#14b8a6', '#65a30d']
LIVE_TRACE_COLORS = ['#f59e0b', '#ec4899', '#f97316', '#eab308', '#fb7185', '#fbbf24']

class fit_plot():
    """Fit graph for dataset-based training, x-axis fixed to the FIRST
    input. Each OUTPUT gets its own pair of traces:
      Data trace (markers) -- that output's raw (first-input x, y) dataset
                               points, set whenever the dataset table
                               changes.
      Live trace (markers) -- that output's ACTUAL live (input, output)
                               pairs, sampled once per forward() tick
                               while Play is running -- not a synthetic
                               sweep -- so what's drawn is a true
                               reflection of what the running network
                               produces, comparable against the Data trace.
    Trace pairs are appended/removed dynamically as outputs are added to
    or removed from the network (see add_output_trace/remove_output_trace)."""

    def __init__(self, id, max_run_points=500):
        self.id = id
        self.max_run_points = max_run_points
        self.output_order = []   # oid, in trace-append order; oid's Data trace is at index 2*pos, Live at 2*pos+1
        self.run_data = {}       # oid -> {"xs": deque, "ys": deque}
        layout = js.JSON.parse(json.dumps({
            "margin": {"l": 36, "r": 14, "t": 10, "b": 30},
            "xaxis": {"tickfont": {"size": 10}, "zeroline": True},
            "yaxis": {"tickfont": {"size": 10}, "autorange": True, "zeroline": True},
            "showlegend": True,
            "legend": {"font": {"size": 10}, "x": 0, "y": 1}
        }))
        Plotly.newPlot(self.id, js.JSON.parse(json.dumps([])), layout)

    def _pos(self, oid):
        return self.output_order.index(oid)

    def add_output_trace(self, oid, label=None):
        if oid in self.output_order:
            return
        pos = len(self.output_order)
        color = DATA_TRACE_COLORS[pos % len(DATA_TRACE_COLORS)]
        live_color = LIVE_TRACE_COLORS[pos % len(LIVE_TRACE_COLORS)]
        name = label or f"y{pos + 1}"
        data_trace = {'x': [], 'y': [], 'mode': 'markers', 'type': 'scatter',
                      'name': f'{name} data', 'marker': {'color': color, 'size': 9}}
        live_trace = {'x': [], 'y': [], 'mode': 'markers', 'type': 'scatter',
                      'name': f'{name} live', 'marker': {'color': live_color, 'size': 6, 'opacity': 0.75}}
        Plotly.addTraces(self.id, js.JSON.parse(json.dumps([data_trace, live_trace])))
        self.output_order.append(oid)
        self.run_data[oid] = {"xs": deque([], self.max_run_points), "ys": deque([], self.max_run_points)}

    def remove_output_trace(self, oid):
        if oid not in self.output_order:
            return
        pos = self._pos(oid)
        Plotly.deleteTraces(self.id, js.JSON.parse(json.dumps([pos * 2, pos * 2 + 1])))
        self.output_order.pop(pos)
        self.run_data.pop(oid, None)

    def update_data_points(self, oid, xs, ys):
        if oid not in self.output_order:
            return
        Plotly.restyle(self.id, js.JSON.parse(json.dumps({'x': [list(xs)], 'y': [list(ys)]})),
                        js.JSON.parse(json.dumps([self._pos(oid) * 2])))

    def reset_run(self):
        for oid in self.output_order:
            rd = self.run_data[oid]
            rd["xs"].clear()
            rd["ys"].clear()
            Plotly.restyle(self.id, js.JSON.parse(json.dumps({'x': [[]], 'y': [[]]})),
                            js.JSON.parse(json.dumps([self._pos(oid) * 2 + 1])))

    def add_run_point(self, oid, x, y):
        rd = self.run_data.get(oid)
        if rd is None:
            return
        rd["xs"].append(x)
        rd["ys"].append(y)
        Plotly.restyle(self.id, js.JSON.parse(json.dumps(
            {'x': [list(rd["xs"])], 'y': [list(rd["ys"])]})), js.JSON.parse(json.dumps([self._pos(oid) * 2 + 1])))


class loss_plot():
    """Simple scrolling line plot for loss-vs-epoch during training."""

    def __init__(self, id, max_points=500):
        self.id = id
        self.max_points = max_points
        self.ys = deque([], max_points)
        self.traces = js.JSON.parse(json.dumps([
            {'y': [], 'mode': 'lines', 'name': 'Loss', 'line': {'color': '#d97706'}}
        ]))
        layout = js.JSON.parse(json.dumps({
            "margin": {"l": 40, "r": 10, "t": 10, "b": 26},
            "xaxis": {"title": {"text": "epoch", "font": {"size": 10}}, "tickfont": {"size": 9}},
            "yaxis": {"tickfont": {"size": 9}, "autorange": True},
            "showlegend": False
        }))
        Plotly.newPlot(self.id, self.traces, layout)

    def reset(self):
        self.ys.clear()
        self.updatePlot(js.JSON.parse(json.dumps({'y': [[]]})))

    def add_point(self, loss):
        self.ys.append(loss)
        return js.JSON.parse(json.dumps({'y': [list(self.ys)]}))

    def updatePlot(self, update):
        Plotly.update(self.id, update)


class scatter_plot():
    """Persistent (x, y) marker plot -- unlike `plot` above (a scrolling
    line over an implicit time axis), points here accumulate and stay put
    so a set of test inputs/targets can be compared against the network's
    fitted curve."""

    def __init__(self, id, max_points=300):
        self.id = id
        self.max_points = max_points
        self.xs = deque([], max_points)
        self.ys = deque([], max_points)
        self.traces = js.JSON.parse(json.dumps([
            {'x': [], 'y': [], 'mode': 'markers', 'type': 'scatter',
             'name': 'Test points', 'marker': {'color': '#82ca9d', 'size': 8}}
        ]))
        layout = js.JSON.parse(json.dumps({
            "margin": {"l": 28, "r": 10, "t": 10, "b": 22},
            "xaxis": {"tickfont": {"size": 9}, "zeroline": True},
            "yaxis": {"tickfont": {"size": 9}, "autorange": True, "zeroline": True},
            "showlegend": False
        }))
        Plotly.newPlot(self.id, self.traces, layout)

    def addPoint(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        return js.JSON.parse(json.dumps({'x': [list(self.xs)], 'y': [list(self.ys)]}))

    def addPoints(self, num, data):
        # Compatibility no-op: an output's device/channel hookup still routes
        # live hardware notifications here (see Device.py's notif_callback),
        # but this plot is meant to only move on explicit addPoint() calls
        # from Test(), so continuous notifications are ignored.
        return js.JSON.parse(json.dumps({}))

    def updatePlot(self, update):
        Plotly.update(self.id, update)

    def clear(self):
        self.xs.clear()
        self.ys.clear()
        self.updatePlot(js.JSON.parse(json.dumps({'x': [[]], 'y': [[]]})))