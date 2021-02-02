# BATCHES allow calculations to be parrallelized. More cores = faster calcs. GPU's have thousands of cores. CPU's have 4, 8, 16, etc.
# BATCHES also help with generalization (fits in-sample data more smoothly. Don't want to overfit, though. Somewhere around batch sizes of 4 8 16 32 etc is usually good.)

import numpy as np

np.random.seed(1)

# input feature sets usally are 'X' in machine learning (training data set)
X = [[1, 2, 3, 3.5], # number of neurons x number of weights
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# define two hidden layers. 'hidden' because we don't really control those layers.
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # 'He initialization (2015)'...'Xavier initialization (2010) would be 1 instead of 2'...both are to prevent the exploding and vansihing gradient problem (non-optimum solutions)
        self.weights = np.random.randn(
            n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# first number is size of first line of inputs/'X'
# second number can be any number we want = size of 1st hidden layer
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2) # first number has to be same as second number of layer 1

layer1.forward(X) # size (3,4)
print(layer1.output) # size ()
layer2.forward(layer1.output)
print(layer2.output)


#########################################################
#### EVERYTHING UNDER HERE IS HARD-CODED...NEED TO GENERALIZE CODE
# # 1ST LAYER
# inputs = [[1, 2, 3, 2.5],
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]]

# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
# biases = [2, 3, 0.5]

# # 2ND LAYER
# weights2 = [[0.1, -0.14, 0.5],
#             [-0.5, 0.12, -0.33],
#             [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# # need to transpost weights for shape congruency in dot product
# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)
#######################################################
