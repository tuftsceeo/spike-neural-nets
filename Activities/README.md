Better Activity Progression:
Level 1: How does a neural network work
Activity 1: Build a car to slow down as it reaches the wall
- Intro to neural network, could not be simpler (1 in, 1 out, 1 layer, bias = [0], weight = [1] (or less if they choose))
- Basic structure, what is a linear layer

Activity 2: OR gate with joystick
- Builds on first, and now you have to think about an activation function and a bias.

Activity 3: XOR gate with joystick
- Builds on previous two, but now we need a hidden layer

Now that you understand the structure (neurons, layers, activation functions) start modeling in pytorch to train it.

Level 2: Training Math
Activity 1: Matrix Multiplication. Visual Calculator. 
- 
- No need for acutal visualization here, I will include a link to elsewhere
Activity 2: Loss - MSE. Visual Calculator. 
- 
Activity 3: Backpropagation - Visualization of plotting different loss from weights and biases, and getting slope. Just to visualize what gradient even means in this context. 
Activity 4: Optimization - Instead of plotting points and drawing the slope, we use calc to find the slope at that one instance (vector), not showing calc, but showing vector and stepping process. Gradient descent activity
Activity 5: Epochs - Graphs of num epochs over time
Activity 6: Learning rate - graph of different rates over time
Activity 7: Shuffling Data - graph of shuffled data vs ordered
Activity 8: Batching Data - graphs of different batch sizes
Activity 9: Train set vs test set - conceptual, just a prompt

Level 3: Translating into Pytorch
Activity 1: Defining the class
Activity 2: Training
- 2.1: Write pseudocode
- 2.2: Translating that into actual code


Original Activity Progression (Way more in the weeds, uses pytorch):
Level 0:
    - Don't even touch PyTorch yet, just build and program a car to slow down as it reaches the wall
    - The speed of the motors should be proportional to the relfection/distance value
    - Draw a map, where you have your input layer (distance/relfection) and your output layer (speed)

Level 1:
Now, instead of hardcoding that relationship, recreate that in PyTorch to have a neural net do that for you. This level is a very basic intro into neural nets, and the core concepts of training them
    - Part 1: Establish the structre
    - Part 2: Write the forwards function
    - Part 3: Write a simple training script
    - Part 4: How does num epochs impact accuracy?
    - Part 5: How does the learning rate impact the output at each epoch?
    - Part 6: Testing your model

Level 2:
Level 1 was about linearity and keeping it very simple, so now we wnat to intro non-linearity
    - Part 1: Establish the structure
    - Part 2: Write the forwards function
    - Part 3: Training: a proper explanation of backpropagation     

Level 3:
Where to go now? More neurons/layers? Different types of layers (ex convolutional)? Different types of non linear functions? Different loss/gradient discent functions?
