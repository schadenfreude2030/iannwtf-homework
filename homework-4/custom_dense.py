import tensorflow as tf


# Custom Layer
class CustomDense(tf.keras.layers.Dense):

    def __init__(self, units=256, activation=tf.nn.sigmoid):
        self.weight = None
        self.bias = None
        super(CustomDense, self).__init__(units, activation=activation)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        drive = tf.matmul(inputs, self.weight) + self.bias
        return self.activation(drive)
