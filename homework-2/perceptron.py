import numpy as np
import functions


class Perceptron:

    def __init__(self, input_units):
        self.weights = np.random.randn(input_units)
        self.bias = np.random.randn()
        self.alpha = 1  # learning rate
        self.latest_weighted_sum = None
        self.latest_activation = None
        self.latest_inputs = None

    def forward_step(self, inputs):
        self.latest_inputs = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.latest_weighted_sum = weighted_sum  # weighted sum: z(L)
        self.latest_activation = functions.sigmoid(weighted_sum)  # activation: a(L)
        return self.latest_activation

    def update(self, changes_weights, changes_bias):
        self.weights = self.weights - changes_weights
        self.bias = self.bias - changes_bias
