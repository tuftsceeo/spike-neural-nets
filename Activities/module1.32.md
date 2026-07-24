# Backpropagtation

Once it has found the loss, how does the model know how to change its thinking?

This is where backpropagation comes in. The model needs to know which parts of its function are making it more wrong, so it backtracks through the equation to find how each part is impacting the output. When it backtracks, it asigns **gradients** to each part of the equation (like the weights and biases). Gradient is just another word for slope, so these gradients are telling the model how much changing one value of the weight changes the output. 

Play around with this to get a sense of how slope impacts the output:

[SIMULATION HERE: it has a slider where you an change the value of the weight and one for the bias. Then plots the loss from that. Slope is how much changing weight or bias changes the loss]