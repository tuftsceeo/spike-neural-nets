import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

class myNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_layer = nn.Linear(1, 1)
    def forward(self, input):
        output = self.main_layer(input)
        return output
    
def train(model, sample_inputs, sample_outputs):
    print("training")
    dataset = TensorDataset(sample_inputs, sample_outputs)
    loader = DataLoader(dataset)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

    for epoch in range(50):
        for input, true_output in loader:
            optimizer.zero_grad()
            guess = model(input)
            loss = criterion(guess, true_output)
            loss.backward()
            optimizer.step()

    return model

def generate_data(num_points=100, x_range=(0, 100)):
    x = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(1)
    y = 3*x + 5 # Say this is the relationship between distance (or relfection) and speed
    return x, y

# Testing it
model = myNeuralNet()
x, y = generate_data()
model = train(model, x, y)

for input in range(10):
    tensorx = torch.tensor(input).float().unsqueeze(0)
    predicted = model(tensorx)
    print("x: " + str(input) + ", y: " + str(predicted))
