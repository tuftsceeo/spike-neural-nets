"""Module-level state shared across the horizontal-backpropagator UI.

Every other module reaches into this one via `import state` and reads/writes
state.x (never `from state import x`) so mutations are visible everywhere --
PyScript's cross-module rebinding only works through attribute access.
"""
from pyscript import document

# Each layer is a single neuron: {id, w, b, act}. The chain is strictly
# linear -- layer i's only input is layer (i-1)'s activation (or the raw
# input x for layer 0).
layers: list[dict] = []
layer_counter = 0

biases_enabled = False
lr = 0.1

# Dataset: [{"id": int, "x": float, "y": float}, ...]
dataset: list[dict] = []
data_counter = 0

ACTIVATION_OPTIONS = [
    ("None", "none"),
    ("ReLU", "relu"),
    ("Leaky ReLU", "leaky_relu"),
    ("Sigmoid", "sigmoid"),
    ("Tanh", "tanh"),
    ("Softplus", "softplus"),
]

# ── Training state machine ───────────────────────────────────────────────
# step_index 0 == ready to run a forward pass. Every layer reveals in TWO
# sub-steps -- first the activation node (dL/dz_i = dL/da_i * da_i/dz_i),
# then the linear/weight node (dL/dw_i = dL/dz_i * dz_i/dw_i) -- so
# step_index runs 1..2*len(layers), plan is still one entry per LAYER
# (ordered from the LAST layer to the FIRST), and substep s belongs to
# plan[(s-1)//2] with odd s = activation, even s = linear/weight.
epoch = 0
step_index = 0
plan: list[dict] = []          # this epoch's full backward plan, computed once at step 0
forward_cache: dict | None = None
loss_history: list[tuple[int, float]] = []
initialized = False
playing = False

# Backward-gradient arrows currently drawn under the diagram, cleared at
# the start of every forward pass. Each entry: {"source_id", "target_id", "html"}.
grad_markers: list[dict] = []

# Snapshot stack for backward step/epoch (see network_model.take_snapshot).
history: list[dict] = []

# Narrower than (-1, 1) -- a deep strict chain of ReLU-family layers is
# very prone to a large early weight pushing a pre-activation negative and
# zeroing the gradient for every layer behind it ("dying"), so random init
# leans smaller to reduce (not eliminate -- that's an inherent property of
# a 1-wide chain) how often that happens.
INIT_WEIGHT_RANGE = (-0.7, 0.7)
MAX_GRAD_NORM = 5.0


def get_id(id_: str):
    return document.getElementById(id_)


def next_layer_id() -> int:
    global layer_counter
    layer_counter += 1
    return layer_counter


def next_data_id() -> int:
    global data_counter
    data_counter += 1
    return data_counter
