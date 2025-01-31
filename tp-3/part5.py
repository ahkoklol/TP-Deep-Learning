import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to include the channel dimension
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input layer with the nome shape
input_layer = tf.keras.layers.Input(shape=(None, None, 1), name="input_layer")

c=128

# Bloc convolutif 1 avec connexion résiduelle reste inchangé car il utilise une convolution 1x1
x = input_layer
input_to_block = x  # Sauvegardez l'entrée pour la connexion résiduelle
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
# Adding the residual
x = x + input_to_block

# Appliquer la même logique pour les autres blocs convolutifs
for i in range(2, 5):  # Répéter pour les blocs 2, 3 et 4
    input_to_block = x
    for _ in range((7-1)//2):  # Utilisez (7-1)//2 pour simuler une convolution 7x7
        x = tf.keras.layers.Conv2D(filters=c, kernel_size=(3, 1), padding="same")(x)
        x = tf.keras.layers.Conv2D(filters=c, kernel_size=(1, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    # Ajoutez la connexion résiduelle
    x = x + input_to_block

conv_layer = x


# Flatten the output of the convolutions
flatten_layer = tf.keras.layers.Flatten()(conv_layer)
# Pas utilisé ici sion le program marche pas

# Nouvelle couche Conv2D avec 10 filtres avant le GlobalAveragePooling
conv_layer = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu', padding='same')(conv_layer)

# GlobalAveragePooling2D
gap_layer = tf.keras.layers.GlobalAveragePooling2D()(conv_layer)

# Add the final activation layer
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(gap_layer)

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

#%%
