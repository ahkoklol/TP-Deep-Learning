import tensorflow as tf
import numpy as np
from tp6_data import OOV_CHAR, WORD_INDEX, INDEX_FROM, INVERTED_WORD_INDEX


class SPE(tf.keras.layers.Layer):
    def __init__(self, output_dims: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_dims = output_dims

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dims": self.output_dims,
            }
        )
        return config

    def call(self, inputs):
        # Get the dynamic sequence length from the inputs
        seq_length = tf.shape(inputs)[1]

        # Compute positional encodings
        position = tf.range(seq_length, dtype=tf.float32)
        freqs = tf.range(self.output_dims // 2, dtype=tf.float32)
        freqs = 1 / (10000 ** (2 * freqs / self.output_dims))

        angles = position[:, tf.newaxis] * freqs[tf.newaxis, :]
        pos_encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)

        # Extend pos_encoding to match batch size
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.tile(pos_encoding, [tf.shape(inputs)[0], 1, 1])


class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size  # Taille du vocabulaire
        self.embed_size = embed_size  # Dimension de l'embedding

    def build(self, input_shape):
        super().build(input_shape)
        # Couche d'embedding pour les tokens
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embed_size
        )
        # Couche d'embedding positionnel
        self.pos_emb = SPE(output_dims=self.embed_size)

    def call(self, inputs):
        token_embedding = self.token_emb(inputs)  # Création de l'embedding des tokens
        pos_embedding = self.pos_emb(inputs)  # Création de l'embedding positionnel
        return token_embedding + pos_embedding  # Somme des embeddings de token et positionnel


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, ff_proj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads  # Le nombre de têtes d'attention
        self.ff_proj = ff_proj  # La dimension de la couche dense du feed-forward

    def build(self, input_shape):
        super().build(input_shape)
        embed_dim = input_shape[-1]  # La dimension des embeddings

        # Couche d'attention multi-têtes
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=embed_dim // self.num_heads
        )

        # Première couche de normalisation
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Couches denses pour le feed-forward network
        self.dense1 = tf.keras.layers.Dense(self.ff_proj, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(embed_dim)

        # Seconde couche de normalisation
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Couche de dropout pour la régularisation
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        # Calcul de l'attention multi-têtes
        attention_path = self.multihead_attention(inputs, inputs, inputs)

        # Première connexion résiduelle et normalisation
        attention_path = self.layer_norm1(inputs + attention_path)

        # Passage par le réseau feed-forward
        dense_path = self.dense1(attention_path)
        dense_path = self.dense2(dense_path)

        # Application du dropout
        dense_path = self.dropout(dense_path)

        # Seconde connexion résiduelle et normalisation
        output = self.layer_norm2(attention_path + dense_path)

        return output



class TextGen(tf.keras.callbacks.Callback):
    def __prompt(self, text, originality):
        print()
        predict_and_write(self.model, prompt=text, size=50, originality=originality)

    def on_epoch_end(self, epoch, logs=None):
        self.__prompt("This movie is", 1)
        self.__prompt("This movie is", 2)
        self.__prompt("This movie is", 5)
        self.__prompt("This movie is", 10)
        self.__prompt("This movie is", 20)
        print()
        self.model.save(f"model_{epoch}")
        return super().on_epoch_end(epoch, logs)


def predict_and_write(model, prompt: str, size: int, originality: int):
    print(prompt, end=" ")
    prompt = [1] + [  # <- start with the starting symbol [START]
        WORD_INDEX.get(p.lower().strip(), OOV_CHAR - INDEX_FROM)
        + INDEX_FROM  # <- change all words into their indexes
        for p in prompt.split(" ")
        if p != " "
    ]
    for _ in range(size - len(prompt) + 1):
        pred = model(np.asarray(prompt)[np.newaxis, ...])
        pred = tf.keras.activations.softmax(pred)[0, -1, ...]
        topk = tf.math.top_k(pred, k=originality)
        i = np.random.choice(topk.indices.numpy(), 1)[0]
        prompt.append(i)
        print(INVERTED_WORD_INDEX[i], end=" ", flush=True)
