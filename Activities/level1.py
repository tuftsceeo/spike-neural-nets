import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
# LEVEL 1 PYTORCH ACTIVITIES. Takes in the color sensor input, outputs the motor speed

# First, lets some terms to understand what it means to pass something through the NN
nn.Linear(inputs, output) # --> this is a shorthand for y = Wx + b, where y is the output vector and x is the input vector. W is the weight matrix (really the meat of the layer, how are we transforming the input), and b is the bias, just something to shift our output by

# Now, lets start with establishing the shape of a simple neural net. How many inputs and how many outputs?
class myNeuralNet(nn.Module): # Our model is a class, and that class is a subclass of the torch "Module"
    def __init__(self) # we want it to do all the setup that the broad "Module" class does
        super().__init__()
        self.main_layer = nn.Linear(1, 1) # takes in 1 input, returns 1 output

    # Now that we have made our NN, let's define what a forward pass looks like. 
    # main_layer is a function, so we want to simply give that layer our input, and get the output
    def forward(self, input):
        output = self.main_layer(input)
        return output

# Now we need to train our model.
# Our model needs examples of what inputs produce what outputs so it knows what it is supposed to do.
# So, we can train it on a labeled data set: inputs that we labled with the correct outputs
def train(model, sample_inputs, sample_outputs):
    # PyTorch needs to have its data as Tensors (basically just matrices, they bind the inputs with the labels)
    # So, lets merge our two samples into one TensorDataset
    dataset = TensorDataset(sample_inputs, sample_outputs)

    # There's one more thing we have to do to our dataset to make it easy to read, it needs to be in a DataLoader
    # This just wraps our dataset in an iterable, or in other words, it makes it easier and faster for PyTorch to access the data points when training
    loader = DataLoader(dataset)

    # Imagine you are studying for a test.
    # You might do a practice problem, check your answer, tweak your understanding, then try the problem again to see if you get the right answer
    # Our model is just doing that over and over again when it's training. 
    # It takes in the data, compares it's output to the correct output, then tweaks the values in its layers, and tries again to see if it got closer

    # First, lets define the tool that helps us decide the "loss" (how far off our answer was from the actual answer)
    criterion = nn.MSELoss() # criterion is a function that we can use as a shorthand for cross entropy loss (an equation that calculates loss)

    # Now, lets define the tool that helps us figure out how to tweak the values in the layer based on that loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    # the optimizer uses Adam to figure out how it needs to update our values
    # Adam uses complicated math, but the key thing to note is the lr --> that is our learning rate, 
    # and it tells Adam if it should tweak the values by a lot or a little every time it has to readjust the values

    # Now we can begin the actual training
    # Remember, our model is basically just retrying practice problems over and over again, so we can tell it how many times it should retry
    # For now, lets say we want it to retry 10 times, or epochs, so we should define a loop that runs for 10 epochs
    for epoch in range(10):
        # In each epoch, it has to run through the dataset. We can think of the dataset as the test, and each
        # data point as a problem in the test
        # For each problem, it has to:
        # 1) try the problem
        # 2) check it's work to see how far off it was
        # 3) figure out how it was wrong
        # 4) fix it's understanding

        for input, true_output in loader:
            #
            optimizer.zero_grad()

            # 1) try the problem
            guess = model(input) # we are using the inputed model here like a function. This calls our forward function

            # 2) check its work by finding the loss
            loss = criterion(guess, true_output)

            # 3) figure out how it was wrong
            # We know our prediction was off by some amount, but now we have to figure out which values in the matrix caused that, and how we should fix them
            # In this case, our matrix is simple. We have a 1 x 1 matrix (number), and a 1 x 1 basis (number). 
            # So, we are in the form of guess = W(input) + b where W and b are numbers
            # We want to know how those numbers are changing the output, and which direction they are shifting it
            # So, we need the slope, or gradient, of this equation, and adjust it accordingly
            # That is what we do below. backwards just means go back through the equation, and figure out what the slope (gradient) is
            loss.backwards()

            # 4) fix its understanding. Use the gradient calculated by loss.backwards(), and make some changes to our equation
            optimizer.step()
    
    return model
    # Thats all we have to do for this simple model, but lets play around and see what happens if we change things up
    # What happens if we change the learning rate? Try a learning rate of 0.00001, and a learning rate of 0.1, and look at the plot. What works and doesn't work for each?
    # What happens if we change the number of epochs? Try different number of epochs and see how our loss changes on the plot. What are the pros and cons of few or many epochs?
    
# What if we want to test our model to see how accurate it is before putting it in the real world?
# Before, we were just retaking the same practice test over and over again.
# This makes us really good at that specific practice test, but how do we know we aren't just
# memorizing the answers, and are actually improving our understanding? How can we feel confident
# before attending the final exam?

# We can instead just save some of the problems from that practice test to practice on much later,
# after we feel confident in the other problems. 
# This is splitting our dataset into a train set and a test set. Lets say we want to have 80% of our data to train on
# and 20% to test on. This is how we can split our dataset before turning it into a DataLoader:
def train(model, sample_inputs, sample_outputs):
    dataset = TensorDataset(sample_inputs, sample_outputs)
    
    train_size   = int(0.8 * len(dataset)) # Define that we want 80% of the data
    train_dataset, test_dataset = torch.utils.data.random_split(
                       dataset, [train_size, len(dataset) - train_size]) # Randomly split the dataset
    
    # Now continue as usual, using our train_dataset instead of the entire dataset
    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for input, true_output in train_loader:
            optimizer.zero_grad()
            guess = model(input)
            loss = criterion(guess, true_output)
            loss.backwards()
            optimizer.step()
    
    # Now, if we want, we can test our model on those test problems that we set aside for later
    # We want to make sure that we save these problems for last, that way we can feel confident that
    # however our model performs on these data points is how it will perform in the real world

    # Lets say we just want to calculate the average difference in our model's predictions and the actual output
    total_difference = 0
    for input, true_output in test_loader:
        # get the guess
        guess = model(input)

        # Find the difference
        difference = abs(guess - true_output)

        # Increment the total difference
        total_difference += difference

    # Calculate and print the average difference
    average_difference = total_difference/len(test_dataset)
    print("The average difference is: " + average_difference)

    return model

# Levels to add:
# Should probably talk about batches in this part to
# Non-linearity: go from a linear function to sin
# torch.relu(output) # --> this is what we can use to make our model nonlinear. If every layer just outputted it's linear prodcut, then we could condense 

