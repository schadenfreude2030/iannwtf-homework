import perceptron
import functions
import numpy as np


class MLP:
    HIDDEN_LAYER_SIZE = 4

    def __init__(self):
        self.hidden_layer = [perceptron.Perceptron(2) for i in range(self.HIDDEN_LAYER_SIZE)]
        self.output_perceptron = perceptron.Perceptron(self.HIDDEN_LAYER_SIZE)
        self.hidden_layer_activations = None
        self.output_activation = None

    def forward_step(self, inputs):
        self.hidden_layer_activations = [layer.forward_step(inputs) for layer in self.hidden_layer]
        self.output_activation = self.output_perceptron.forward_step(self.hidden_layer_activations)
        return self.output_activation

    def backprop_step(self, expected_output):
        loss = (self.output_activation - expected_output) ** 2
        delta_z_over_w = 2 * (self.output_activation - expected_output)
        delta_a_over_z = functions.sigmoid_prime(self.output_perceptron.latest_weighted_sum)
        delta_C_over_a = np.array(self.hidden_layer_activations)
        changes_weights = (delta_z_over_w * delta_a_over_z * delta_C_over_a)
        changes_bias = (delta_z_over_w * delta_a_over_z)
        self.output_perceptron.update(
            changes_weights,
            changes_bias
        )

        hidden_weights_changes = [
            sum([
                self.output_perceptron.weights[i] * ptron.latest_activation for i in range(self.HIDDEN_LAYER_SIZE)
            ]) * functions.sigmoid_prime(ptron.latest_weighted_sum) * np.array(ptron.latest_inputs) for ptron in self.hidden_layer
        ]
        hidden_bias_changes = [
            functions.sigmoid_prime(ptron.latest_weighted_sum) for ptron in self.hidden_layer
        ]
        for i in range(self.HIDDEN_LAYER_SIZE):
            self.hidden_layer[i].update(
                hidden_weights_changes[i],
                hidden_bias_changes[i]
            )
        return loss
