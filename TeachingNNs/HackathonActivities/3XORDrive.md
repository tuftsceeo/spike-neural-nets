# XOR Controlled Car

## Goal

Now, do the same as the last activity, but instead of using an OR gate, use an XOR (exclusive or). So:
- Either of the joysticks being active drives the car at the default speed
- Both of the joysticks being active does not move the car
- Both the joysticks being inactive does not move the car

Again, try drawing your diagram before diving into the coding, and think about how you can use activation functions. See tips for some hints.

## Tips

<details>
<summary>Read Tips</summary>

- You might have trouble modeling this problem with just one equation, so think about how you can break it into smaller problems, each with their own equation
    - Bonus hint: In particular, can you describe XOR in terms of two simpler logic gates? (Try looking up NAND...)
- How would you draw your diagram now that you broke it down some? Think about the flow of inputs to outputs, and add some extra circles if you need to.

</details>

## The Diagram
<details>
<summary>Solution and Further Thinking</summary>

Hopefully, you got a diagram that looks something like this:

![XOR Diagram](activity3_diagramcopy.jpeg)

This is just like a real neural network: it has a linear input layer, a linear hidden layer, and a linear output layer, and uses activation functions to handle non-linearity.

Also, pay attention to the flow of information. Every node in an input layer connects to every node in it's output layer, that allows to to represent this in matricies. Try turning this network a series of matrix equations: y = Wx + b. (If you need a quick intro to matrix multiplication, play around with this calculator: http://matrixmultiplication.xyz/). For example, we could translate a diagram like this as so:

![Diagram to matrices explainer](diagram_to_matrices.jpeg)

Being able to write every neurons equation in a layer as just one matrix makes life much simpler for us when programming how to use those equations.

Try writing your XOR diagram as a series of equations with weight matrices and bias vectors. 

</details>