INIT_RANGE = (-1, 1)

# ─────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────

class TrainingState:
    def __init__(self):
        self.p1 = (0.0, 0.0)
        self.p2 = (0.0, 0.0)
        self.error_key = "mse"
        self.custom_error_expr = ""
        self.lr = 0.1

        self.w = 0.0
        self.b = 0.0
        self.epoch = 0

        self.forward_result = None
        self.grad_result = None
        self.update_result = None

        self.loss_history = []
        self.step_index = 0
        self.initialized = False
        self.playing = False

        self.last_w_dir = None
        self.last_b_dir = None

        # Snapshot stack for backward step/epoch. Each entry fully
        # describes the state right after some do_step() call finished.
        self.history = []


state = TrainingState()
