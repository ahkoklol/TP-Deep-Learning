import tensorflow as tf
import numpy as np

# Load the MNIST dataset from TensorFlow Keras datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the image data to values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input layer with the shape of MNIST images
input_layer = tf.keras.layers.Input(name="input_layer", shape=(28, 28))

# Flatten the 2D input data to 1D for processing
flatten_layer = tf.keras.layers.Flatten()(input_layer)

# Add a dense (fully connected) layer with 128 neurons
hidden_layer = tf.keras.layers.Dense(units=128)(flatten_layer)

# Apply ReLU activation function to the hidden layer
hidden_layer = tf.keras.layers.Activation("relu")(hidden_layer)

# Apply dropout to prevent overfitting, dropping 20% of the units
hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)

# Add another dense layer for the final classification
hidden_layer = tf.keras.layers.Dense(units=10)(hidden_layer)

# Apply softmax activation for multi-class classification
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(hidden_layer)

# Create the model defining the input and output layers
model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

# Print a summary of the model
model.summary(150)

# Compile the model with Adam optimizer, loss function, and accuracy metric
model.compile(
    optimizer="Adam",
    loss={"output_layer": tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics=["acc"],
)

# Train the model on training data and validate on test data
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=5
)
