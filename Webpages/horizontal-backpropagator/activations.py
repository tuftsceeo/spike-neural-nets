"""Pure activation-function math -- no DOM, no state mutation."""
import math

LEAKY_RELU_SLOPE = 0.01


def apply_activation(x: float, fn: str) -> float:
    if fn == "relu":
        return max(0.0, x)
    elif fn == "leaky_relu":
        return x if x > 0 else LEAKY_RELU_SLOPE * x
    elif fn == "sigmoid":
        return 1.0 / (1.0 + math.exp(-x))
    elif fn == "tanh":
        return math.tanh(x)
    elif fn == "softplus":
        return math.log(1.0 + math.exp(x))
    else:
        return x


def apply_activation_derivative(pre_act: float, post_act: float, fn: str) -> float:
    """d(post_act)/d(pre_act) -- the local slope backprop needs."""
    if fn == "relu":
        return 1.0 if pre_act > 0 else 0.0
    elif fn == "leaky_relu":
        return 1.0 if pre_act > 0 else LEAKY_RELU_SLOPE
    elif fn == "sigmoid":
        return post_act * (1.0 - post_act)
    elif fn == "tanh":
        return 1.0 - post_act ** 2
    elif fn == "softplus":
        return 1.0 / (1.0 + math.exp(-pre_act))
    else:
        return 1.0


ACTIVATION_SYMBOL = {
    "none": "",
    "relu": "ReLU",
    "leaky_relu": "Leaky ReLU",
    "sigmoid": "σ",
    "tanh": "tanh",
    "softplus": "softplus",
}
