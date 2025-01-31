import tensorflow as tf


"""Helper custom layers"""
class ResidualConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), activation='relu', **kwargs):
        super(ResidualConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation(activation)
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation(activation)

        # Adaptation layer to match dimensions
        self.adapt_conv = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Adapt input dimensions if necessary
        adapted_inputs = self.adapt_conv(inputs) if inputs.shape[-1] != self.filters else inputs

        return tf.keras.layers.add([x, adapted_inputs])

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config


class UNetEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(UNetEncoder, self).__init__(**kwargs)
        self.residual_block = ResidualConv2D(filters)
        self.downsample = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')

    def call(self, inputs):
        residual_output = self.residual_block(inputs)
        downsampled_output = self.downsample(residual_output)
        return downsampled_output, residual_output

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config

    def build(self, input_shape):
        super().build(input_shape)
        bs, w, h, c = input_shape

class UNetDecoder(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(UNetDecoder, self).__init__(**kwargs)
        self.upsample = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.concat = tf.keras.layers.Concatenate()
        self.residual_block = ResidualConv2D(filters)

    def call(self, inputs, skip_connection):
        x = self.upsample(inputs)
        x = self.concat([x, skip_connection])  # Merge with skip connection
        x = self.residual_block(x)
        return x





