import math

from state import state

# ─────────────────────────────────────────────────────────────────
# Pure math
# ─────────────────────────────────────────────────────────────────

def _mae_grad(pred, y):
    if pred == y:
        return 0.0
    return 1.0 if pred > y else -1.0


ERROR_FUNCTIONS = {
    "mse": (lambda pred, y: (pred - y) ** 2, lambda pred, y: 2 * (pred - y)),
    "mae": (lambda pred, y: abs(pred - y), _mae_grad), # abs pred - y is gradient
}

SAFE_NAMES = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": lambda x: math.sqrt(x) if x >= 0 else float("nan"),
    "exp": math.exp,
    "log": lambda x: math.log(x) if x > 0 else float("nan"),
    "sin": math.sin,
    "cos": math.cos,
}


def safe_eval(expr, pred, y):
    ns = dict(SAFE_NAMES)
    ns["pred"] = pred
    ns["y"] = y
    return eval(expr, {"__builtins__": {}}, ns)


def custom_error_fn(expr):
    def e(pred, y):
        return safe_eval(expr, pred, y)

    def grad(pred, y):
        h = 1e-4
        return (e(pred + h, y) - e(pred - h, y)) / (2 * h)

    return e, grad


def resolve_error_fns():
    if state.error_key == "custom":
        return custom_error_fn(state.custom_error_expr)
    return ERROR_FUNCTIONS[state.error_key]


def error_label():
    return {"mse": "MSE", "mae": "MAE", "custom": "Custom"}[state.error_key]


def forward(x1, y1, x2, y2, w, b):
    return {"pred1": w * x1 + b, "pred2": w * x2 + b}


def compute_error_and_grads(x1, y1, pred1, x2, y2, pred2, e_fn, grad_fn):
    e1, e2 = e_fn(pred1, y1), e_fn(pred2, y2)
    E = (e1 + e2) / 2
    g1, g2 = grad_fn(pred1, y1), grad_fn(pred2, y2)
    dE_dw = (g1 * x1 + g2 * x2) / 2
    dE_db = (g1 + g2) / 2
    return {"E": E, "e1": e1, "e2": e2, "dE_dw": dE_dw, "dE_db": dE_db}


def direction_arrow(delta):
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def compute_update(w, b, dE_dw, dE_db, lr):
    delta_w, delta_b = -lr * dE_dw, -lr * dE_db # i flipped the sign so that a positive delta would mean that it goes up
    return {
        "delta_w": delta_w,
        "delta_b": delta_b,
        "w_new": w + delta_w,
        "b_new": b + delta_b,
        "w_dir": direction_arrow(delta_w),
        "b_dir": direction_arrow(delta_b),
    }


ARROW_GLYPH = {"up": "▲", "down": "▼", "flat": "–"}


def fmt(x, n=3):
    try:
        return f"{x:.{n}f}"
    except (ValueError, TypeError):
        return str(x)
