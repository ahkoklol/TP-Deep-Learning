import tensorflow as tf
import numpy as np
from ResidualEDSequentialySeparatedConv2D import ResidualEDSequentialySeparatedConv2D

# Load the CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input layer with the shape of CIFAR-10 images
input_layer = tf.keras.layers.Input(shape=(32,32,3), name="input_layer")

# The number of filters and the encoded size can be adjusted according to the model's needs
original_c = 128  # The original number of filters
encoded_c = 32    # The reduced number of filters for encoding


x = tf.keras.layers.Conv2D(filters=original_c, kernel_size=(1, 1), padding='same')(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)

# Apply the same logic for the other convolutional blocks
# Appliquer la même logique pour les autres blocs convolutifs
for i in range(2, 5):  # Répéter pour les blocs 2, 3 et 4
    x = ResidualEDSequentialySeparatedConv2D(kernel_size=7,projection_filter=32)(x)

conv_layer = tf.keras.layers.Conv2D(filters=10, kernel_size=(7,7), activation='relu', padding='same')(x)


# GlobalAveragePooling2D instead of Flatten
gap_layer = tf.keras.layers.GlobalAveragePooling2D()(conv_layer)

# Final activation layer
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(gap_layer)

# Create the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer="Adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=5)
