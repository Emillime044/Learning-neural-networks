import math
import random

def sigmoid(x):
    return 1/(1+math.exp(-x))

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
expected_results = [1, 0, 0, 0, 0, 0, 0, 1]
learning_rate = 0.5
weights_l2 = [0.3, 0.9, 0.6]
bias_l2 = 0.4
outputs = []
neuron_number = int(input("Insert the number of neurons: "))
num_inputs = len(inputs[0])
weights = []
for n in range(neuron_number):
    neuron_weights = []
    for x in range(num_inputs):
        neuron_weights.append(random.uniform(-1, 1))
    weights.append(neuron_weights)

biases = []
for n in range (neuron_number):
    biases.append(random.uniform(-1, 1))

weights_l2 = []
for n in range(neuron_number):
    weights_l2.append(random.uniform(-1, 1))

for cycle in range(10000):
    for i in range(len(inputs)):
        layer1_outputs = []
        # Layer 1 forward pass
        for n in range(neuron_number):
            z = 0
            for x in range(len(inputs[i])):
                z += inputs[i][x] * weights[n][x]
            z += biases[n]
            layer1_outputs.append(sigmoid(z))
        
        # Layer 2 forward pass
        z = 0
        for x in range(neuron_number):
            z += layer1_outputs[x] * weights_l2[x]
        z += bias_l2
        output = sigmoid(z)    

        # Layer 2 backprop
        error = output - expected_results[i]
        loss = (output - expected_results[i])**2

        final_delta = delta(error, output)
        for x in range(neuron_number):
            gradient = final_delta*layer1_outputs[x]
            weights_l2[x] = weights_l2[x] - learning_rate * gradient            
        bias_l2 = bias_l2 - learning_rate * final_delta

        for n in range(len(inputs[i])):
            # Layer 1 backprop
            blame = final_delta*weights_l2[n]
            neuron_delta = blame * derivative_sigmoid(layer1_outputs[n])
            for x in range(neuron_number):
                gradient = neuron_delta * inputs[i][x]
                weights[n][x] = weights[n][x] - learning_rate * gradient
            biases[n] = biases[n] - learning_rate * neuron_delta

for i in range(len(inputs)):
    layer1_outputs = []
    for n in range(neuron_number):
        z = 0
        for x in range(len(inputs[i])):
            z += inputs[i][x] * weights[n][x]
        z += biases[n]
        layer1_outputs.append(sigmoid(z))
    z = 0
    for x in range(neuron_number):
        z += layer1_outputs[x] * weights_l2[x]
    z += bias_l2
    print(inputs[i], "->", sigmoid(z))