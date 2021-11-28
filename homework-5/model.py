import tensorflow as tf
import numpy as np


def calc_accuracy(pred, target):
    """Calculate accuracy between a prediction and a target.

    :param pred: a prediction that the model made
    :type pred: tf.Tensor of floats
    :param target: target that model should have predicted
    :type target: tf.Tensor of floats
    """
    same_prediction = tf.argmax(target, axis=1) == tf.argmax(pred, axis=1)
    return np.mean(same_prediction)


class MyCnnModel(tf.keras.Model):
    """This is a custom model class

        :param loss_function: loss function used to calculate loss of the model
        :type loss_function: function from the tf.keras.losses module
        :param optimizer: optimizer used to apply gradients to the models
            trainable variables
        :type optimizer: function from the tf.keras.optimizers module
        """

    def __init__(self, loss_function, optimizer):
        super(MyCnnModel, self).__init__()
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.layers_array = [
            # concrete layer structure is result of try and error
            # 2 conv layers and one max pooling operation
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                   activation="relu", padding='valid'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                   activation="relu", padding='valid'),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same'),
            # another 2 conv layers and a global average pool operation
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                   activation="relu", padding='valid'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                   activation="relu", padding='valid'),
            tf.keras.layers.GlobalAveragePooling2D(),
            # dense layer of 10 because there are 10 target classes
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute the feed-forward pass through all dense layers.

        :param inputs: network input
        :type inputs: tf.Tensor
        """
        x = inputs
        for layer in self.layers_array:
            x = layer(x)
        return x

    def train_step(self, input: tf.Tensor, target: tf.Tensor) -> (float, float):
        """Applys optimizer to all trainable variables of this model to
        minimize the loss (loss_function) between the target output and the
        predicted ouptut.

        :param input: input to the model
        :type input: tf.Tensor
        :param target: target output with repect to the input
        :type target: tf.Tensor
        :return: the loss and the accuracy of the models prediction
        :rtype: tuple of two floats
        """
        with tf.GradientTape() as tape:
            prediction = self(input)
            loss = self.loss_function(target, prediction)
            gradients = tape.gradient(loss, self.trainable_variables)
        # apply gradients to the trainable variables using a optimizer
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        accuracy = calc_accuracy(prediction, target)
        return loss, accuracy

    def test(self, test_data: tf.data.Dataset):
        """Calculate the mean loss and accuracy of the model over all elements
        of test_data.

        :param test_data: model is evaulated for test_data
        :type test_data: tensorflow 'Dataset'
        :return: mean loss and mean accuracy for all datapoints
        :rtype: tuple of two floats
        """
        # aggregator lists for tracking the loss and accuracy
        test_accuracy_agg = []
        test_loss_agg = []
        # iterate over all input-target pairs in test_data
        for (input, target) in test_data:
            prediction = self(input)
            loss = self.loss_function(target, prediction)
            accuracy = calc_accuracy(prediction, target)
            # add loss and accuracy to aggregators
            test_loss_agg.append(loss.numpy())
            test_accuracy_agg.append(np.mean(accuracy))
        # calculate mean loss and accuracy
        test_loss = tf.reduce_mean(test_loss_agg)
        test_accuracy = tf.reduce_mean(test_accuracy_agg)
        return test_loss, test_accuracy

