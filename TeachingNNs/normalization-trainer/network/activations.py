"""Pure activation-function math -- no DOM, no state mutation. Every
function here takes all its inputs as arguments."""
import math
import re
import traceback
import numpy as np

from state import CUSTOM_ACT_NAMES

IMPLICIT_MULT_RE = re.compile(r'(?<=[0-9])(?=[a-zA-Z(])|(?<=[a-zA-Z)])(?=[0-9(])')

def normalize_expr(expr: str) -> str:
    cleaned = expr.replace("^", "**")
    cleaned = IMPLICIT_MULT_RE.sub("*", cleaned)
    return cleaned

def safe_eval_expr(expr: str, x: float) -> float:
    if not expr or not expr.strip():
        return x
    cleaned = normalize_expr(expr)
    ns = dict(CUSTOM_ACT_NAMES)
    ns["x"] = x
    try:
        return float(eval(cleaned, {"__builtins__": {}}, ns))
    except Exception:
        print("Custom activation eval error:\n" + traceback.format_exc())
        return x

def parse_bound(raw: str):
    if raw is None:
        return None
    s = raw.strip().lower().replace("∞", "inf")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def apply_custom_activation(x: float, custom_activation: dict) -> float:
    pieces = custom_activation["pieces"]
    if not pieces:
        return safe_eval_expr(custom_activation["expr"], x)
    for p in pieces:
        lo, hi = parse_bound(p["lo"]), parse_bound(p["hi"])

        if lo is None:
            lo_ok = True
        elif p["lo_op"] == "<":
            lo_ok = lo < x
        elif p["lo_op"] == "<=":
            lo_ok = lo <= x
        elif p["lo_op"] == ">":
            lo_ok = lo > x
        elif p["lo_op"] == ">=":
            lo_ok = lo >= x
        else:
            lo_ok = True

        if hi is None:
            hi_ok = True
        elif p["hi_op"] == "<":
            hi_ok = x < hi
        elif p["hi_op"] == "<=":
            hi_ok = x <= hi
        elif p["hi_op"] == ">":
            hi_ok = x > hi
        elif p["hi_op"] == ">=":
            hi_ok = x >= hi
        else:
            hi_ok = True

        if lo_ok and hi_ok:
            return safe_eval_expr(p["expr"], x)
    return 0.0

def sigmoid_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))

def apply_activation(x: float, fn: str, custom_activation: dict | None = None) -> float:
    if fn == "relu":
        return max(0.0, x)
    elif fn == "sigmoid":
        return sigmoid_numpy(x)
    elif fn == "tanh":
        return math.tanh(x)
    elif fn == "softplus":
        return math.log(1.0 + math.exp(x))
    elif fn == "custom":
        return apply_custom_activation(x, custom_activation or {"expr": "x", "pieces": []})
    else:
        return x

def apply_activation_derivative(pre_act: float, post_act: float, fn: str,
                                 custom_activation: dict | None = None) -> float:
    """d(post_act)/d(pre_act), used by backpropagate() for the chain rule."""
    if fn == "relu":
        return 1.0 if pre_act > 0 else 0.0
    elif fn == "sigmoid":
        return post_act * (1.0 - post_act)
    elif fn == "tanh":
        return 1.0 - post_act ** 2
    elif fn == "softplus":
        return sigmoid_numpy(pre_act)
    elif fn == "custom":
        h = 1e-4
        ca = custom_activation or {"expr": "x", "pieces": []}
        return (apply_custom_activation(pre_act + h, ca) - apply_custom_activation(pre_act - h, ca)) / (2 * h)
    else:
        return 1.0
