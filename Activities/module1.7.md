# Training in Batches
What if instead of taking one practice problem at a time before checking your work, you took 10 problems and then checked your work and updated your understanding of the concept? What might be the benefits of our model training with that approach, rather than the one at a time approach?

Make a prediction, then play with this simulation:

[insert link to simulation that allows you to change the batch size, and side by side look at what is happening]

<details>
<summary>Continue</summary>

One thing you might have noticed is that it reduces the **noise** of the data. Instead of being skewed largely by one weird datapoint, it can look at many at once get a better sense of what direction to take.

## How to Batch in PyTorch
We can tell our DataLoader to batch the data for us very simply by adding one thing to our dataloader declaration:

```python
loader = DataLoader(train_ds, batch_size=16, shuffle=True)
```

The rest of our training function can stay exactly the same.

</details>
