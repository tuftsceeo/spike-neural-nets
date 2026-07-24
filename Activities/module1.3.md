# Labled Datasets
While there are many ways to train a neural network, we will train ours by using supervised learning. This means we will give it a labled data set: a list of inputs, and the correct corresponding outputs. 

So, with our wall stopping example, say we are using the equation,
```python
speed = 1/2 * reflection
```
Write a 5 point labled dataset that we could feed to our model. What is the input? What is the label? Then look at the example below.

<details>
<summary>Continue</summary>

For example, we could train our model on a larger version of the following dataset:
| Reflection | Speed    |
| -----------| -------- |
| 0 | 0 |
| 20 | 10 |
| 40 | 20 |
| 60 | 30 |
| 80 | 40 |

</details>



