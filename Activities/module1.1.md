# Establishing The Structure

## What is a Neural Network?

[insert explanation]

## Building That In PyTorch

Go back to your problem from module 0. You programmed your car to get slower as it nears the wall.
You had one input (distance) and one output (speed). To get from one to the other, you just multiplied your distance by some constant to get the speed, which is a linear equation that looked something like this:
```python
speed = c*distance
```
Note that we could have had a b value (like in y = mx + b), but becuase our y intercept is 0 for this equation, we just use b = 0.

Based on what you know about neural networks, write down how many layer(s) we need, and how any inputs and outputs each of those layer(s) should have. Then continue below to see the solution.

<details>
<summary>Continue</summary>

If each layer of a neural network is a linear equation in the form of y = Wx + b, we can use just one layer with one input and one output for this. In PyTorch, we can define linear layers as such:
```python
import torch.nn as nn
layer = nn.Linear(num_inputs, num_outputs)
```
With this in mind, try writing a WallStopNetwork class that has your desired number of layers as parameters, each having the correct inputs and outputs. These things shold be set up within the class's constructor. We also want to make sure we tell PyTorch that this class is a neural network, so make your class extend nn.Module and use nn.Module's constructor within your constructor. Then compare your solution to the one below.

<details>
<summary>Continue</summary>
We can write this class as so:

```python
import torch.nn as nn
class WallStopNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layer1 = nn.Linear(1, 1)
```
</details>
</details>

