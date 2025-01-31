import tensorflow as tf

inputs_shape=(32, 32, 3)

class MyCustomLayer(tf.keras.layers.Layer):

    # Define the constructor
    def __init__(self, custom_parameter=42, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_parameter = custom_parameter

    # Define the build method
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=inputs_shape)

    # Define the call method
    def call(self, inputs):
        # Define the forward computation using inputs and self.kernel
        return inputs + self.kernel

    # Define the get_config method
    def get_config(self):
        config = super().get_config()
        config.update({
            # must have the same name as the self. method
            "custom_parameter": self.custom_parameter
        })
        return config
