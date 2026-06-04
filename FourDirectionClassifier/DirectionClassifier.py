"""
DirectionClassifier.py — Defines, trains, and extracts weights from the classifier.

Responsibilities:
  - Generating the labeled dataset
  - Defining and training the neural network
  - Extracting weights as plain Python lists for use elsewhere
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# Neural network definition
# ══════════════════════════════════════════════════════════════════════════════

class DirectionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset generation
# ══════════════════════════════════════════════════════════════════════════════

def _label_angle(angle):
    """
    Map an angle (0–360°) to a direction class.
    Order matches SPIKE motor positioning (0° = up):
      Up=0, Right=1, Down=2, Left=3
    """
    angle = angle % 360
    if angle < 45 or angle >= 315:
        return 0  # Up
    elif angle < 135:
        return 1  # Right
    elif angle < 225:
        return 2  # Down
    else:
        return 3  # Left


def _build_dataset(num_samples=1000):
    angles_deg = np.random.uniform(0, 360, num_samples)
    labels     = np.array([_label_angle(a) for a in angles_deg])
    angles_rad = np.deg2rad(angles_deg)
    X = np.column_stack([np.sin(angles_rad), np.cos(angles_rad)])
    return (
        torch.tensor(X,      dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train() -> DirectionClassifier:
    """Train the classifier and return the trained model."""
    X_tensor, y_tensor = _build_dataset()

    dataset      = TensorDataset(X_tensor, y_tensor)
    train_size   = int(0.8 * len(dataset))
    train_ds, _  = torch.utils.data.random_split(
                       dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model     = DirectionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training model...")
    for epoch in range(30):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30")

    print("Training done.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Weight extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_weights(model: DirectionClassifier) -> dict:
    """Return all layer weights and biases as plain Python lists (no torch)."""
    def to_list(tensor):
        return tensor.detach().numpy().tolist()

    return {
        "w1": to_list(model.fc1.weight),   # shape (16, 2)
        "b1": to_list(model.fc1.bias),     # shape (16,)
        "w2": to_list(model.fc2.weight),   # shape (16, 16)
        "b2": to_list(model.fc2.bias),     # shape (16,)
        "w3": to_list(model.fc3.weight),   # shape (4, 16)
        "b3": to_list(model.fc3.bias),     # shape (4,)
    }