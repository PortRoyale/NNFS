import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# make sure weights are first, so dimensions (shape) aling for dot product
# output = np.dot(inputs, weights) + biases
output = np.dot(weights, inputs) + biases
print(output)




# ## WITHOUT NUMPY DOT PRODUCT
# layer_outputs = [] # output of current layer
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0 # output of given neuron
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight # add up all inputs * weights
#     neuron_output += neuron_bias # finally, tack on the neuron's bias term
#     layer_outputs.append(neuron_output) # append and save each layer output
# print(layer_outputs)


# ## random example of what weights and biases do
# some_value = -0.5 # random input value
# weight = 0.7 # multiplier of value
# bias = 0.7 # offsets value
# print(some_value*weight)
# print(some_value + bias)


