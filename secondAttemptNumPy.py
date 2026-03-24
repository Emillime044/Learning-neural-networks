import math
import random
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return x*(1-x)

def delta(error, output):
    return error*derivative_sigmoid(output)

# Hardcoded inputs for learning AND logic

inputs = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]
inputs = np.array(inputs)
expected_results = [1, 0, 0, 0, 0, 0, 0, 1]
expected_results = np.array(expected_results)
cycle_count = int(input("Insert the number of cycles you want to do: "))
learning_rate = 0.1
weights_l2 = [0.3, 0.9, 0.6]
bias_l2 = 0.4
outputs = []
neuron_number = int(input("Insert the number of neurons: "))
num_inputs = len(inputs[0])
weights = np.random.uniform(-1, 1, (neuron_number, num_inputs))
biases = np.random.uniform(-1, 1, neuron_number)
weights_l2 = np.random.uniform(-1, 1, neuron_number)

for cycle in range(cycle_count):
    for i in range(len(inputs)):
        layer1_outputs = []
        # Layer 1 forward pass
        layer1_outputs = sigmoid(np.dot(inputs[i], weights.T) + biases)
        
        # Layer 2 forward pass
        z = np.dot(layer1_outputs, weights_l2) + bias_l2
        output = sigmoid(z)    

        # Layer 2 backprop
        error = output - expected_results[i]
        loss = (output - expected_results[i])**2

        final_delta = delta(error, output)
        weights_l2 = weights_l2 - learning_rate * final_delta * layer1_outputs
        bias_l2 = bias_l2 - learning_rate * final_delta

        # Layer 1 backprop
        blame = final_delta * weights_l2
        neuron_deltas = blame * derivative_sigmoid(layer1_outputs)
        
        weights = weights - learning_rate * np.outer(neuron_deltas, inputs[i])
        biases = biases - learning_rate * neuron_deltas

for i in range(len(inputs)):
    layer1_outputs = sigmoid(np.dot(inputs[i], weights.T) + biases)
    z = np.dot(layer1_outputs, weights_l2) + bias_l2
    print(inputs[i], "->", sigmoid(z))
print("\nFinal weights:")
print("Layer 1 weights:", weights)
print("Layer 1 biases:", biases)
print("Layer 2 weights:", weights_l2)
print("Layer 2 bias:", bias_l2)
