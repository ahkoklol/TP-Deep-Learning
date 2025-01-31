import tensorflow as tf
import numpy as np

START_CHAR = 1
OOV_CHAR = 2
INDEX_FROM = 2

WORD_INDEX = tf.keras.datasets.imdb.get_word_index()
INVERTED_WORD_INDEX = dict((i + INDEX_FROM, word) for (word, i) in WORD_INDEX.items())

INVERTED_WORD_INDEX[START_CHAR] = "[START]"
INVERTED_WORD_INDEX[OOV_CHAR] = "[OOV]"
INVERTED_WORD_INDEX[0] = "[PAD]"


def load_data(vocab_size, batch_size):
    (data, _), (_, _) = tf.keras.datasets.imdb.load_data(
        num_words=vocab_size,
        oov_char=OOV_CHAR,
        start_char=START_CHAR,
        index_from=INDEX_FROM,
    )
    # Finding the length of the longest sequence
    max_len = max(len(sequence) for sequence in data)

    # Padding the sequences
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(
        data, maxlen=max_len, padding="post", truncating="post"
    )

    np_array = np.array(padded_data)

    x_tr = np_array[:, :-1]
    x_te = np_array[:, 1:]

    text_ds = tf.data.Dataset.from_tensor_slices((x_tr, x_te))
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)
    text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

    return text_ds
