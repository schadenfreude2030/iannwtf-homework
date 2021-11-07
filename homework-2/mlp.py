import perceptron
import functions
import numpy as np


class MLP:
    HIDDEN_LAYER_SIZE = 4

    def __init__(self):
        self.hidden_layer = [perceptron.Perceptron(2) for i in range(self.HIDDEN_LAYER_SIZE)]
        self.output_perceptron = perceptron.Perceptron(self.HIDDEN_LAYER_SIZE)
        self.epsilon = 1
        self.last_inputs = None
        self.hidden_layer_activations = None
        self.output_activation = None

    def forward_step(self, inputs):
        # save all activations for backprop step
        self.last_inputs = inputs
        self.hidden_layer_activations = np.array([layer.forward_step(inputs) for layer in self.hidden_layer])
        self.output_activation = self.output_perceptron.forward_step(self.hidden_layer_activations)
        return self.output_activation

    def backprop_step(self, expected_output):
        # calc loss for return
        loss = (self.output_activation - expected_output) ** 2

        # delta for output perceptron
        output_activation = np.array([self.output_activation])
        delta = (expected_output - output_activation) * (output_activation * (1 - output_activation))
        self.output_perceptron.update(self.hidden_layer_activations, delta, self.epsilon)

        # delta for hidden layer
        hidden_activation = self.hidden_layer_activations
        weights_matrix = np.array([self.output_perceptron.weights[:-1]])  # without bias
        sigmoid_prime = hidden_activation * (np.ones_like(hidden_activation) - hidden_activation)
        delta = (delta.T @ weights_matrix) * sigmoid_prime

        for i in range(self.HIDDEN_LAYER_SIZE):
            self.hidden_layer[i].update(self.last_inputs, delta[i], self.epsilon)

        return loss
