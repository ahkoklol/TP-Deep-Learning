import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to include the channel dimension
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input layer with the new shape
input_layer = tf.keras.layers.Input(shape=(28, 28, 1), name="input_layer")

# Replace the first dense layer with a convolutional layer
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(input_layer)

# Add 3 more convolutional layers as per the instructions
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), activation='relu', padding='same')(conv_layer)
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), activation='relu', padding='same')(conv_layer)
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), activation='relu', padding='same')(conv_layer)

# Flatten the output of the convolutions
flatten_layer = tf.keras.layers.Flatten()(conv_layer)

# Use a dense layer to reduce to the output dimensions needed for classification
hidden_layer = tf.keras.layers.Dense(units=10)(flatten_layer)

# Add the final activation layer
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(hidden_layer)

# Create the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer="Adam",
              loss={"output_layer": tf.keras.losses.SparseCategoricalCrossentropy()},
              metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=5)
