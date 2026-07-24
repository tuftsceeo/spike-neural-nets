# Writing the Train Function

Write just the declaring line of a function called train that takes a model, a list of inputs, and a list of their labled outputs.

<details>
<summary>Continue</summary>
It might look something like this:

```python
def train(model, sample_inputs, sample_outputs):
```

## Formatting the data

PyTorch needs to have its data as Tensors, which are basically just matrices. Here, they just connect the input values with the labeled output values like a table.

```python
dataset = TensorDataset(sample_inputs, sample_outputs)
```

It then needs to turn that dataset into a DataLoader, which just makes it easier for the program to access the datapoints.

```python
loader = DataLoader(dataset)
```

## Writing the Loop

Now, lets define some tools to help with this process.

Our loss calculator using Mean Squared Error (MSE):
```python
 criterion = nn.MSELoss()
``` 

Next, we need a way to know how to change our network to reduce that loss value.
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
``` 
The optimizer uses an something called Adam to figure out how it needs to update the values in the network. Adam does **gradient descent**, which is a way of trying to find the minimum of a function, in this case that function is the loss function. By tracking all the gradients calculated using backpropagation, it can see how changing those gradients will minimize the loss. 

There are a couple things to note when using PyTorch's Adam:
1) One key thing is that we input the learning rate, lr, which tells the optimizer how much it should tweak the values in the network by each time it checks its work.
2) Because Adam is constantly checking and storing all those gradients, we have to make sure to reset it before each time that we move on to a new datapoint, otherwise all of the previous gradients that it stored for other data points will make it confused. This process of zeroing the gradients can be done with:
    ```python
    optimizer.zero_grad()
    ```

Try writing a function to train the model that does the following:
Sets up the dataset
Sets up the training tools.
Goes through the data set 10 times (or epochs), and in each time:
    Go through each datapoint. For each point:
        1) Before doing anything, reset the optimizer
        2) Pass it through the network. *(Hint: Remember we can pass something through using model(input))*
        3) Find the loss *(Hint: criterion is a function that can take in an estimated output and a true output)*
        4) Figure out what caused that loss *(Hint: use loss.backwards(), we will go over this later)*
        5) Adjust the model accordingly *(Hint: calling optimizer.step() will adjust the weights and biases as needed)*
Returns the model that we trained

When you have tried that, continue below to see the solution

<details>
<summary>Continue</summary>
Based on that outline, our function should look something like this:

```python
def train(model, sample_inputs, sample_outputs):
    # Setup the dataset
    dataset = TensorDataset(sample_inputs, sample_outputs)

    loader = DataLoader(dataset)

    # Setup the training tools
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Go through the dataset 10 times
    for epoch in range(10):

        # Review each datapoint
        for input, true_output in loader:

            # Reset the optimizer
            optimizer.zero_grad()

            # Get the model's current estimate of an input
            guess = model(input)

            # Calculate the loss by comparing it against the true output
            loss = criterion(guess, true_output)

            # Figure out what contributed to that loss using loss.backwards()
            loss.backwards()

            # Change our network accordingly
            optimizer.step()
    # Return the trained model
    return model
```
And that's all we need to train our model. Next, we'll put it to the test, and see how changing certain variables changes our model's training process.

</details>
</details>