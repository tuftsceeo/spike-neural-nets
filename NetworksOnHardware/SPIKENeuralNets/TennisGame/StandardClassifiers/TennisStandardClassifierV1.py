import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import csv

# ══════════════════════════════════════════════════════════════════════════════
# Neural network definition
# ══════════════════════════════════════════════════════════════════════════════

class TennisClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(180, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)
        self.verbose = False

    def forward(self, x):
        if self.verbose:
            print("Before pass:")
            print(x)
        x = torch.relu(self.fc1(x))
        if self.verbose:
            print("After fc1:")
            print(x)
        x = torch.relu(self.fc2(x))
        if self.verbose:
            print("After fc2:")
            print(x)
        x = self.fc3(x)
        if self.verbose:
            print("After fc3:")
            print(x)
        return x

LABELS = ["Forehand", "Backhand", "Overhead", "None"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "gesture_data")

# ══════════════════════════════════════════════════════════════════════════════
# Dataset generation
# ══════════════════════════════════════════════════════════════════════════════

def _build_list(filename, X_list, y_list):
    # Determine class index from filename
    gesture_name = filename.removesuffix(".csv")
    label = LABELS.index(gesture_name)

    path = f"{DATA_DIR}/{filename}"

    with open(path, newline="") as f:
        reader = csv.reader(f)

        # Skip header row
        next(reader)
        for row in reader:
            # Convert all 180 values to floats
            values = [float(v) for v in row]
            # make sure there are 180 values
            if len(values) > 180:
                del values[-(len(values) - 180):]

            X_list.append(values)
            y_list.append(label)

def _build_dataset():
    X_list, y_list = [], []

    _build_list("Forehand.csv", X_list, y_list)
    _build_list("Backhand.csv", X_list, y_list)
    _build_list("Overhead.csv", X_list, y_list)
    _build_list("None.csv", X_list, y_list)

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train() -> TennisClassifer:
    """Train the classifier and return the trained model."""
    X_tensor, y_tensor = _build_dataset()

    dataset      = TensorDataset(X_tensor, y_tensor)
    train_size   = int(0.8 * len(dataset))
    train_ds, _  = torch.utils.data.random_split(
                       dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model     = TennisClassifer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training model...")
    for epoch in range(30): # 30 is arbitrary, just saying that we want to pass the data through 30 times (ie check itself against data 30 times)
        model.train() # put in training mode (doesnt do much here really)
        total_loss = 0
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
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/30  Loss: {total_loss/len(train_loader):.4f}")

    print("Training done.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Weight extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_weights(model: TennisClassifer) -> dict:
    """Return all layer weights and biases as plain Python lists (no torch)."""
    def to_list(tensor):
        return tensor.detach().numpy().tolist()

    return {
        "w1": to_list(model.fc1.weight),  
        "b1": to_list(model.fc1.bias),
        "w2": to_list(model.fc2.weight),
        "b2": to_list(model.fc2.bias),
        "w3": to_list(model.fc3.weight),
        "b3": to_list(model.fc3.bias),
    }

def set_weights(model: TennisClassifer, w1, b1, w2, b2, w3, b3):
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor(w1, dtype=torch.float32))
        model.fc1.bias.copy_(torch.tensor(b1, dtype=torch.float32))
        model.fc2.weight.copy_(torch.tensor(w2, dtype=torch.float32))
        model.fc2.bias.copy_(torch.tensor(b2, dtype=torch.float32))
        model.fc3.weight.copy_(torch.tensor(w3, dtype=torch.float32))
        model.fc3.bias.copy_(torch.tensor(b3, dtype=torch.float32))