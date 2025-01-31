import tensorflow as tf
from EDSequentialySeparatedConv2D import EDSequentialySeparatedConv2D


class ResidualEDSequentialySeparatedConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size,projection_filter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.projection_filter = projection_filter

    def call(self, inputs):
        input_to_block = inputs
        for layer in self.layers:
            inputs = layer(inputs)
        inputs = inputs + input_to_block
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "projection_filter": self.projection_filter
        })
        return config

    def build(self, input_shape):
        super().build(input_shape)
        bs, w, h, c = input_shape
        #c correspond au nombre de filtres de base de la couche précédente
        self.layers = []

        self.layers.append(EDSequentialySeparatedConv2D(kernel_size=self.kernel_size,projection_filter=self.projection_filter))


