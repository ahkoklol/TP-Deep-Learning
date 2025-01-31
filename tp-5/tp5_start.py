import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from data_generator import SegmentationDataGenerator
from tp5_utils import ResidualConv2D, UNetEncoder, UNetDecoder

BATCHSIZE = 4
DEPTH = 4
PROJ_SIZE = 8
NCLASSES = 1


"""Model Code"""
# input
inputs = tf.keras.layers.Input(shape=(None, None, 3), name="images")

# transform
hidden_layer = tf.keras.layers.Conv2D(filters=PROJ_SIZE, kernel_size=(1, 1))(inputs)
hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
hidden_layer = tf.keras.layers.Activation("relu")(hidden_layer)
hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)

# Here goes the UNET CODE:

# U-Net Architecture
skip_connections = []
for depth in range(DEPTH):
    hidden_layer, skip = UNetEncoder(PROJ_SIZE * (2 ** depth))(hidden_layer)
    skip_connections.append(skip)
# Bottleneck
hidden_layer = ResidualConv2D(PROJ_SIZE * (2 ** DEPTH))(hidden_layer)

# Decoder
for depth in reversed(range(DEPTH)):
    skip_connection = skip_connections.pop()
    hidden_layer = UNetDecoder(PROJ_SIZE * (2 ** depth))(hidden_layer,skip_connection)


# transform back
hidden_layer = tf.keras.layers.Conv2D(filters=NCLASSES, kernel_size=(1, 1))(
    hidden_layer
)
hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

# predict
hidden_layer = tf.keras.layers.Activation("sigmoid", name="output")(hidden_layer)

model = tf.keras.models.Model(inputs, hidden_layer)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    # Special Cross-Entropy Loss Function for when there is only 2 classes:
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics="acc",
)

IMAGE_PATH = glob("dataset/images/*.jpg")
ANNOT_PATH = glob("dataset/labels/*.jpg")

data_generator_train = SegmentationDataGenerator(
    IMAGE_PATH[:40],
    ANNOT_PATH[:40],
    BATCHSIZE,
    DEPTH,
)
data_generator_test = SegmentationDataGenerator(
    IMAGE_PATH[40:],
    ANNOT_PATH[40:],
    1,
    DEPTH,
)

model.fit(
    data_generator_train,
    validation_data=data_generator_test,
    epochs=1000,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            "val_acc",
            patience=50,
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=100,
        )
    ],
)

model.evaluate(data_generator_test)

model.save("model")

#%%

#%%
