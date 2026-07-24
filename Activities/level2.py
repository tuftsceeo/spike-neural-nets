import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

# LEVEL 2 PYTORCH ACTIVITIES: Making it non-linear

# Before, our neural network was just like a linear equation. We took in a value, multiplied it by a slope, added a bias, and returned
# But, what if we wanted to make it non-linear? What if we could not just get the answer by multiplying and adding something to our input
# This much more closely aligns with real world scenarios, where neural networks have to model complicated relationships between inputs and outputs
# Lets take a non-linear equation: y = x^2
# We can build and train our model very similarly, we just have to make a few key changes
class myNeuralNet(nn.Module):
    # We know we still want one input and one output, but what about what is in between?
    # Before, we just had one layer (equation) that was transforming our input by a slope and a bias, so it was linear
    # However, this needs to be non-linear. What if we added a second layer, that took the output from the first layer,
    # and produced an output of its own? 
    # In our second layer, we can have more inputs, or neurons, to make our network more complex.
    # Say we wanted our first layer to take in one input (x), produce 4 outputs to feed into 4 neurons in the next layer, and that layer takes in those 4 neurons and outputs 1 y value
    layer1 = nn.Linear(1, 4) # 1 input, 4 outputs
    layer2 = nn.Linear(1, 1) # 4 inputs, 1 output

    # The issue with this, though, is that we are just plugging one linear function into another which is still linear:
    # y = W2(W1*x + b) + b --> y = (W1*W2)*x + (W2*b + b)
    # So, how can we introduce some non-linearity between these two layers?
    # For that, we can use a number of functions, but one is called ReLU.
    # ReLU is incredibly simple, all it does is it turns negative inputs into 0, and positive numbers stay the same
    # While it is simple, it is enough non-linearity to basically just shake up our model, and allow it to leave a linear boundry
    # We can implement this non-linearity in our forward function
    def forward(self, x):
        # feed it through the first layer
        x = self.layer1(x)

        # introduce the non linearity as ReLU
        x = nn.ReLU(x)

        # feed it through the second layer
        x = self.layer2(x)

        # Return that final result
        return x
    
# Now that we have established what our network looks like and functions, we can train it
# The training function will look exactly the same as before, but, we should think about how it's process is different now
# In particular, how is loss.backwards different?
def train(model, sample_inputs, sample_outputs):
    # The steps for getting our data and turning it into a tensor is all the same:
    # All we are doing is turning into a special datatype (tensor), splitting that tensor up into a dataset to test on and one to train on, and then turning that into another special datatype (DataLoader)
    dataset = TensorDataset(sample_inputs, sample_outputs)

    train_size   = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
                       dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset)

    # We still establish our training tools the same way. PyTorch handles all the complicated math within these functions,
    # so we can establish them the same way
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # We can still run it through 10 times
    for epoch in range(10):

        # We still want to train against each data point in our training dataset
        for input, true_output in train_loader:

            # We still want to zero out all the gradients 
            optimizer.zero_grad()

            # We know based on our forwards function, this will be a little different. An extra couple steps
            guess = model(input)

            # This is a little different, but we do not need to worry about it
            loss = criterion(guess, true_output)

            # This is where things get a bit more complicated
            loss.backwards()
            # We know our equation looks about like this:
            # y = layer2(relu(layer1(x)))    

            # What loss.backwards() is trying to do is work backwards through the equation
            # to see how the loss changes by changing the weights, or what the slope of the loss/weight is for each weight
            
            # The equation for loss is:
            # loss = true_value - our_equation(x)
            # Lets say we want to see how the weights for layer 1 (W1) are impacting the loss
            # We want to differentiate the loss equation with respect to W1
            # dl/dW1 = d/dW1(true_value - (our_equation(x))) = d/dW1(our_equation(x))

            # so, to get dl/dW1, we just have to take the derivative of our equation with respect to W1:
            # y = layer2(relu(layer1(x)))
            # To do this, we use the chain rule:
            # dy/dW1 = layer2'(relu(layer1(x))) * relu'(layer1(x)) * layer1'(x), taking all derivatives with respect to W1
            # Now we have found how W1 is changing our output, so we can repeat this for all of the weights and biases in our equation
            # This is why it is called backpropagation. 
            # By using the chain rule to calculate how changing the weights changes our equation, we are going backwards through the equation

            # Now, we can pass off those gradients we calculated to our optimizer 
            # (we dont have to do the passing, PyTorch stores that data with our model, so the optimizer has access to it) 
            # to adjust the weights and biases accordingly
            # So, this is roughly the same, but we have more weights and biases to increment and more gradients to base that off of
            optimizer.step()
    
    return model
