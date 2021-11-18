from custom_dense import *


# Custom Model
class CustomModel(tf.keras.Model):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.hidden_layer_1 = CustomDense(units=32)
        self.hidden_layer_2 = CustomDense(units=32)
        self.output_layer = CustomDense(units=1)

    def call(self, inputs):
        acti = self.hidden_layer_1(inputs)
        acti = self.hidden_layer_2(acti)
        acti = self.output_layer(acti)
        return acti
