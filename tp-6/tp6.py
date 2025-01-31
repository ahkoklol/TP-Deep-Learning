import tensorflow as tf

from tp6_utils import Embeddings, TransformerBlock, TextGen
from tp6_data import load_data


def build_GPT(vocab_size, embed_size, num_heads, num_trblocks, ff_proj):
    inputs = tf.keras.layers.Input(shape=(None,), name="input_indexes")

    # Couches d'embeddings
    x = Embeddings(vocab_size, embed_size)(inputs)

    # Ajout des blocs transformateurs
    transformer_blocks = []
    for _ in range(num_trblocks):
        x = TransformerBlock(num_heads, ff_proj)(x)
        transformer_blocks.append(x)


    # Couche finale pour la prédiction
    hidden_layers = tf.keras.layers.Concatenate()(transformer_blocks)

    outputs = tf.keras.layers.Dense(units=vocab_size, name="out_prediction")(hidden_layers)

    return tf.keras.models.Model([inputs], [outputs])

VOCAB_SIZE = 2000  # Only consider the top vocab_size words
EMBED_SIZE = 128  # Embedding size for each token
NUM_HEADS = 2  # Number of attention heads
NUM_TRAN_BLOCKS = 1  # Number of transformer blocks
LAST_FF_PROJECTION_DIM = (
    2 * EMBED_SIZE
)  # projection size for the last part of transformers
BATCH_SIZE = 64

EPOCHS = 50

text_ds = load_data(VOCAB_SIZE, BATCH_SIZE)

model = build_GPT(
    VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_TRAN_BLOCKS, LAST_FF_PROJECTION_DIM
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.fit(text_ds, epochs=EPOCHS, callbacks=[TextGen()])

model.save("model")
