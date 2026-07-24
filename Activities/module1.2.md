# Defining a Forward Pass

Before, we defined our class like this:
```python
import torch.nn as nn
class WallStopNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layer1 = nn.Linear(1, 1)
```

This was how we established the structure of our network. Now we want to tell PyTorch how it should use this layer when we want to feed an input through the network.

Because this layer is just shorthand for the function y = Wx + b, we just want to pass any input x into the layer1 function:
```python
import torch.nn as nn
class WallStopNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layer1 = nn.Linear(1, 1)

    def forwards(self, x):
        return self.layer1(x)
```

Make sure this function is defined as forwards() as a method of the class. Even though we defined this function, we won't ever call it directly, and we will instead call model(x) to pass x through our model. PyTorch will then call our forwards() function behind the scenes whenever we call model(x).