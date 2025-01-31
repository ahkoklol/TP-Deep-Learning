import tensorflow as tf
class SequentialySeparatedConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config

    def build(self, input_shape):
        super().build(input_shape)
        bs, w, h, c = input_shape
        self.layers = []
        for _ in range((self.kernel_size - 1) // 2):
            self.layers.append(
                tf.keras.layers.Conv2D(filters=c, kernel_size=(3, 1), padding="same")
            )
            self.layers.append(
                tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 3), padding="same")
            )
            self.layers.append(tf.keras.layers.Activation("relu"))
            self.layers.append(tf.keras.layers.Dropout(0.2))
