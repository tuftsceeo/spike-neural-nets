import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import csv
import os

class GestureClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer takes in the 6 args per time stamp, spits out 32. 32 is arbitrary number
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=5)
        
        # Next layer breaks down into 64, 
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)

        # Fully conneted layer to classify the conv layer data
        # SIMPLIFIED BY REMOVING FC1
        self.fc = nn.Linear(64, 4) # 4 output classes

    def forward(self, x):
        x = x.permute(0, 2, 1) # x starts as (batch, 30, 6) and is permuted into (batch, 6, 30) (batch, channels, time)
        x = torch.relu(self.conv1(x)) # pass through first conv layer
        x = torch.relu(self.conv2(x)) # pass through second conv layer
        
        x = x.mean(dim=2) # global average pooling. Average the 22 time stampes into one value to make it 2d instead of 3d
        # pass through fully connected layer
        x = self.fc(x)

        return x

LABELS = ["Forehand", "Backhand", "Overhead", "None"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "gesture_data")


def _build_list(filename, X_list, y_list):
    # Determine class index from filename
    gesture_name = filename.removesuffix(".csv")
    label = LABELS.index(gesture_name)

    path = f"{DATA_DIR}/{filename}"

    with open(path, newline="") as f:
        reader = csv.reader(f)

        # Skip header row
        next(reader)
        r_idx = 0
        for row in reader:
            # Convert all 180 values to floats
            values = [float(v) for v in row]
            if len(values) > 180:
                print("\nFOUND ISSUE AT " + filename + " ON LINE " + str(r_idx) + " IT HAS " + str(len(values)) + " ITEMS")
                del values[-(len(values) - 180):]
                print("\nFIXED, NOW HAS " + str(len(values)) + " ITEMS")

            # break into little 6 float chunks (each time stamp)
            sample = [
                values[i:i+6]
                for i in range(0, len(values), 6)
            ]

            X_list.append(sample)
            y_list.append(label)
            r_idx += 1

def _build_dataset():
    X_list, y_list = [], []

    _build_list("Forehand.csv", X_list, y_list)
    _build_list("Backhand.csv", X_list, y_list)
    _build_list("Overhead.csv", X_list, y_list)
    _build_list("None.csv", X_list, y_list)

    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    return X, y

def train() -> GestureClassifer:
    # set up data
    X_tensor, y_tensor = _build_dataset()
    dataset = TensorDataset(X_tensor, y_tensor)

    # split into batches to train on
    train_size = int(0.8*len(dataset))
    train_ds, _ = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # create the model and training helpers
    model = GestureClassifer()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    for epoch in range(30): # 30 is arbitrary, just saying that we want to pass the data through 30 times (ie check itself against data 30 times)
        model.train() # put in training mode, not really necessary
        total_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad() # reset the optimizer (forget all the gradients)
            loss = criterion(model(Xb), yb) # compute CEL between model output of Xb and actual yb
            loss.backward() # backpropagate on loss tensor
            optimizer.step() # Change the weights as we need to
            total_loss += loss.item()

            # Print how its going every once in a while
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/30  Loss: {total_loss/len(train_loader):.4f}")

    return model

def extract_weights(model: GestureClassifer) -> dict:
    # to return everything as lists to give to spike
    def to_list(tensor):
        return tensor.detach().numpy().tolist()
    
    return {
        "wc1": to_list(model.conv1.weight),
        "bc1": to_list(model.conv1.bias),
        "wc2": to_list(model.conv2.weight),
        "bc2": to_list(model.conv2.bias),
        "wf": to_list(model.fc.weight),
        "bf": to_list(model.fc.bias),
    }


