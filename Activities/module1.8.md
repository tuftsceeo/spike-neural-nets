# Testing the Model

One thing that is common to do is it set aside a chunk of the dataset not to train on. Why do this?

```python
train_size   = int(0.8 * len(dataset)) # training size is 80% of dataset size
train_dataset, no_train_dataset  = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size]) # randomly split the dataset into two parts, one that is the train size and the other that is the rest of the datapoints
```

<details>
<summary>Continue</summary>
Say that after taking your practice test, you want a final understanding of how you will want to perform on your exam. If you tried to get that understanding just by retaking the test you studied with, you would probably get all the questions right, just because you already knew the answers. But if you set aside some of those problems just for a final check, then they can serve as a good prediction for how you will perform on problems you have not seen before.

The same thing applies to our model. We might want to test it with unseen data before we send it out into the real world.

## In PyTorch
You already saw how we split the dataset:
```python
train_size   = int(0.8 * len(dataset))
train_dataset, no_train_dataset  = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
```

To use these two datasets separately, we can do the following:
```python
def train(model, sample_inputs, sample_outputs):
    dataset = TensorDataset(sample_inputs, sample_outputs)

    # Split the data
    train_size   = int(0.8 * len(dataset))
train_dataset, no_train_dataset  = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # use the correct training data to make the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for input, true_output in train_loader: # make sure to use the right dataloader
            optimizer.zero_grad()
            guess = model(input)
            loss = criterion(guess, true_output)
            loss.backwards()
            optimizer.step()
    
    # After training, test the model on the rest of the data, and calculate the average difference
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

```
</details>