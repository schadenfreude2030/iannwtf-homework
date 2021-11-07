import numpy as np
import functions


class Perceptron:

    def __init__(self, input_units):
        self.weights = np.random.normal(size=input_units + 1)
        self.drive = None

    def forward_step(self, inputs):
        inputs = np.append(inputs, 1)  # bias
        self.drive = self.weights @ inputs
        return functions.sigmoid(self.drive)

    def update(self, activations, delta, epsilon):
        activations = np.append(activations, 1)  # bias
        self.weights += epsilon * delta * activations
