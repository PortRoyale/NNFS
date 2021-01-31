import numpy as np

np.random.seed(1)

# input feature sets usally are 'X' in machine learning (training data set)
X = [[1, 2, 3, 3.5],  # number of neurons x number of weights
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# define two hidden layers. 'hidden' because we don't really control those layers.


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# first number is size of first line of inputs/'X'
# second number can be any number we want = size of 1st hidden layer
layer1 = Layer_Dense(4, 5)
# first number has to be same as second number of layer 1
layer2 = Layer_Dense(5, 2)

layer1.forward(X)  # size (3,4)
print(layer1.output)  # size ()
layer2.forward(layer1.output)
print(layer2.output)