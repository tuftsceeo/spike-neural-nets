import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_layer = nn.Linear(1, 1)

    def forward(self, input):
        output = self.main_layer(input)
        return output

def generate_data(num_points=100, x_range=(0, 100)):
    x = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(1)
    y = x*x
    return x, y

# --- Training with live animation ---

def train_animated(model, sample_inputs, sample_outputs, num_epochs=50, lr=0.01):
    dataset = TensorDataset(sample_inputs, sample_outputs)
    loader = DataLoader(dataset)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    epochs_so_far = []

    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training loss over epochs")
    line, = ax.plot([], [], color="steelblue", linewidth=2)
    loss_text = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                        ha="right", va="top", fontsize=10, color="gray")

    plt.tight_layout()
    plt.ion()
    plt.show()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for input, true_output in loader:
            optimizer.zero_grad()
            guess = model(input)
            loss = criterion(guess, true_output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        epochs_so_far.append(epoch + 1)

        # Dynamically rescale y-axis to fit the curve
        ax.set_ylim(0, max(loss_history) * 1.15)

        line.set_data(epochs_so_far, loss_history)
        loss_text.set_text(f"Epoch {epoch + 1}  |  Loss: {avg_loss:.4f}")

        plt.pause(0.05)

    plt.ioff()
    plt.show()

    return model, loss_history


# --- Usage ---

x, y = generate_data()

model = MyNeuralNet()
trained_model, history = train_animated(model, x, y, num_epochs=200, lr=0.01)
