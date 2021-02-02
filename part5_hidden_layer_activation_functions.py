import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# input feature sets usally are 'X' in machine learning (training data set)
X = [[1, 2, 3, 3.5],  # number of neurons x number of weights
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


X, y = spiral_data(100, 3)
print(X, y)
# Note: need atleast 2 hidden layers to model a non-linear problem

# n_inputs = number of inputs
# n_neurons = number of outputs/ number of neurons of next layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # 'He initialization (2015)'...'Xavier initialization (2010) would be 1 instead of 2'...both are to prevent the exploding and vansihing gradient problem (non-optimum solutions)
        self.weights = np.random.randn(
            n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



##### DATA SET MAKER from nnfs library
# #https://cs231n.github.io/neural-networks-case-study/
# def spiral_data(points, classes):
#     X = np.zeros((points*classes, 2))
#     y = np.zeros(points*classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points)  # radius
#         t = np.linspace(class_number*4, (class_number+1)*4,
#                         points) + np.random.randn(points)*0.2
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X, y
#
# import matplotlib.pyplot as plt 
# print("here")
# X, y = spiral_data(100, 3)
# plt.scatter(X[:,0], X[:,1])
# plt.show()
# plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
# plt.show()

