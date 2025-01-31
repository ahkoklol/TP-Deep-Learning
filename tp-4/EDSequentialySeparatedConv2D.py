import tensorflow as tf
from SequentialySeparatedConv2D import SequentialySeparatedConv2D

class EDSequentialySeparatedConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size,projection_filter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.projection_filter = projection_filter

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
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

        self.layers.append(tf.keras.layers.Conv2D(filters=self.projection_filter, kernel_size=(1, 1), padding='same'))
        self.layers.append(tf.keras.layers.BatchNormalization())
        self.layers.append(tf.keras.layers.Activation("tanh"))
        self.layers.append(tf.keras.layers.Dropout(0.2))

        self.layers.append(SequentialySeparatedConv2D(kernel_size=self.kernel_size))

        self.layers.append(tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 1), padding='same'))
        self.layers.append(tf.keras.layers.BatchNormalization())
        self.layers.append(tf.keras.layers.Activation("tanh"))
        self.layers.append(tf.keras.layers.Dropout(0.2))


