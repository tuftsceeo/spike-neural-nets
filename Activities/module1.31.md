# The Basic Training Loop
Imagine you are studying for a test. How would you use a practice test to study for the exam?

First, you might try one problem. Then, you'll check your work, and then you'll tweak your understanding of the concept based on how you were wrong. Then, you'll move onto the next problem and repeat.

Think about how we can apply this to training a neural network, then continue below.

<details>
<summary>Continue</summary>

The training loop for the neural network is just like this, but instead doing problems from a practice test, it calculates an outputs from inputs in the labled dataset, and compares it's outputs to the labeled outputs. And, instead of taking the test (going through the dataset) just once, it will take that test over and over again, improving it's understanding score each time. 

So, our steps should be:

1) Input the datapoint, and see what we get
2) See how far off that guess is from the actual value
3) Figure out where it went wrong
4) Update its understanding accordingly

Then repeat.

</details>