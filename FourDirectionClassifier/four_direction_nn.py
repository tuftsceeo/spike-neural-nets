import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

# Create labeled dataset
num_samples = 1000
angles_deg =  np.random.uniform(0, 360, num_samples) # create 1000 random angles

def label_angle(angle):
    angle = angle % 360
    if angle < 45 or angle >= 315:
        return 0  # up
    elif angle < 135:
        return 1  # right
    elif angle < 225:
        return 2  # down
    else:
        return 3  # left

labels = np.array([label_angle(a) for a in angles_deg])

# Convert each angle to (sin, cos)
angles_rad = np.deg2rad(angles_deg)
X = np.column_stack([np.sin(angles_rad), np.cos(angles_rad)]) # 2 by 1000 array
y = labels

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Train model
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# feed 32 samples at a time during training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# Define neural network
class DirectionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = DirectionClassifier()
print(model)
print(f"\nTotal trainable parameters: "
      f"{sum(p.numel() for p in model.parameters())}\n")


# Loss function and optimizer
criterion = nn.CrossEntropyLoss() # punishes low scores on correct class
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # adjusts weights to reduce loss

# Training Loop
# Each "epoch" is one full pass through the training data.
# Each batch runs the 5-step cycle:
#   1. Clear old gradients
#   2. Forward pass (make predictions)
#   3. Compute loss (measure how wrong we are)
#   4. Backprop (figure out which weights caused the error)
#   5. Update weights (nudge them to reduce loss)

num_epochs = 30
print("Training")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        # Occasionally print loss to make sure avg_loss is decreasing
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1:2d}/{num_epochs}  |  Loss: {avg_loss:.4f}")

# Evaluate on test set
print("/nEvaluating on test set")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predicted = outputs.argmax(dim=1) # index of highest score
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
print(f"Test Accuracy: {correct}/{total} = {100 * correct / total:.1f}%")

# Try it on specific angles
class_names = ["Up", "Right", "Down", "Left"]
 
def predict_angle(angle_deg):
    rad = np.deg2rad(angle_deg)
    x = torch.tensor([[np.sin(rad), np.cos(rad)]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        scores = model(x)
        predicted = scores.argmax(dim=1).item()
    return class_names[predicted]
 
print("\nSample predictions:")
test_angles = [0, 45, 90, 135, 180, 225, 270, 315, 359]
for angle in test_angles:
    print(f"  {angle:>3}° → {predict_angle(angle)}")



    



