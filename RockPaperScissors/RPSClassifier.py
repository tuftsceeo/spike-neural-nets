"""
RPSClassifier.py — Defines, trains, and extracts weights from the classifier.

Responsibilities:
  - Generating the labeled dataset
  - Defining and training the neural network
  - Extracting weights as plain Python lists for use elsewhere
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random
import math


# ══════════════════════════════════════════════════════════════════════════════
# Neural network definition
# ══════════════════════════════════════════════════════════════════════════════

class RPSClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset generation
# ══════════════════════════════════════════════════════════════════════════════

def _finger_open(angle):
    return (angle >= 335 and angle <= 360) or (angle >= 0 and angle <= 20)

def _label_point(fingers):
    """
    Map the three angles to a hand position
    """
    f0, f1, f2 = _finger_open(fingers[0]), _finger_open(fingers[1]), _finger_open(fingers[2])

    if not f0 and not f1 and not f2:
        return 0  # Rock
    elif f0 and f1 and not f2:
        return 2  # Scissors
    elif f0 and f1 and f2:
        return 1  # Paper
    else:
        return -1  # Ambiguous, skip

def _sample_open_angle():
    """Return a random angle in the 'open' range (315–360 or 0–10)."""
    if random.random() < 0.5:
        return random.uniform(335, 360)
    else:
        return random.uniform(0, 20)

def _sample_closed_angle():
    """Return a random angle in the 'closed' range (10–315)."""
    return random.uniform(21, 334)

def _build_dataset(num_samples_per_class=1000):
    """
    Generate a balanced dataset with equal samples per class.
    Each sample is 6 features: (sin, cos) for each of three finger angles.
      Rock     (label 0): all three fingers closed
      Scissors (label 1): fingers 0 and 1 open, finger 2 closed
      Paper    (label 2): all three fingers open
    """
    X_list, y_list = [], []

    for label in range(3):
        for _ in range(num_samples_per_class):

            if label == 0:    # Rock — all closed
                fingers = [_sample_closed_angle() for _ in range(3)]
            elif label == 2:  # Scissors — first two open, third closed
                fingers = [_sample_open_angle(),
                           _sample_open_angle(),
                           _sample_closed_angle()]
            else:             # Paper — all open
                fingers = [_sample_open_angle() for _ in range(3)]

            features = []
            for angle in fingers:
                rad = math.radians(angle)
                features.extend([math.sin(rad), math.cos(rad)])

            X_list.append(features)
            y_list.append(label)

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train() -> RPSClassifier:
    """Train the classifier and return the trained model."""
    X_tensor, y_tensor = _build_dataset()

    dataset      = TensorDataset(X_tensor, y_tensor)
    train_size   = int(0.8 * len(dataset))
    train_ds, _  = torch.utils.data.random_split(
                       dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model     = RPSClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training model...")
    for epoch in range(30): # 30 is arbitrary, just saying that we want to pass the data through 30 times (ie check itself against data 30 times)
        model.train() # put in training mode (doesnt do much here really)
        for Xb, yb in train_loader:
            optimizer.zero_grad() # reset the optimizer (forget all the gradients)
            loss = criterion(model(Xb), yb) # compute CEL between model output of Xb and actual yb
            # Back propagation:
                # the loss is the difference between actual and expected
                # So, it is a function of all the weights before it, just continuously plugged into the next weight
                # And, the gradient (dloss/dweight) is the slope the equation of loss to weight
                # So, gradient says how changing weight affects the loss
                # Then use that to adjust the weight to lower the loss
                # To do it for every weight, we go backwards using the chain rule
                # The gradients are all just derivatives, and each time we plug the output of a weight into a new weight function
                # So, we can use the chain rule to solve for each gradient
            loss.backward() # backpropagate on loss tensor
            
            # Change the weights as we need to
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30")

    print("Training done.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Weight extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_weights(model: RPSClassifier) -> dict:
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