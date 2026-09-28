"""Original DYNAMAP architecture, shared by the country training notebooks.

PoliticalCoords is also reused by the intermediate choice model. Its free
parameters are transformed by a sigmoid into coordinates between zero and one.
Use TensorFlow 2.15.x to retain tracking of these tf.Variable attributes.
"""

import tensorflow as tf


class PoliticalCoords(tf.keras.layers.Layer):
    """Look up coordinates for [proposal A, participant, proposal B] indices."""

    def __init__(self, n_options, n_users, dimension, seed=101):
        super().__init__(name="political_coords")
        self.dimension = dimension
        self.wp = tf.Variable(
            tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=seed)(
                shape=(n_options, dimension), dtype="float32"
            ),
            trainable=True,
            name="proposal_coordinates",
        )
        self.wu = tf.Variable(
            tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=seed)(
                shape=(n_users, dimension), dtype="float32"
            ),
            trainable=True,
            name="participant_coordinates",
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, "int32")
        # One shared proposal table gives each proposal the same point in every choice.
        za = tf.nn.embedding_lookup(self.wp, inputs[:, :1])[:, 0, :]
        zu = tf.nn.embedding_lookup(self.wu, inputs[:, 1:2])[:, 0, :]
        zb = tf.nn.embedding_lookup(self.wp, inputs[:, 2:])[:, 0, :]
        # Concatenate in the order consumed by either choice head, then bound coordinates.
        return tf.nn.sigmoid(tf.concat([za, zu, zb], axis=1))

    def returnOptionsEmbedding(self, withActivation=True):
        return tf.nn.sigmoid(self.wp)

    def returnUsersEmbedding(self, withActivation=True):
        return tf.nn.sigmoid(self.wu)


class NOMOriginal(tf.keras.layers.Layer):
    """Four shared branches per alternative: exp((z_p - z_u)^2 W + b)."""

    def __init__(self, dimension, seed=101):
        super().__init__(name="nom_original")
        self.dimension = dimension
        self.w = tf.Variable(
            tf.random_normal_initializer(seed=seed, stddev=0.05)(
                shape=(dimension, 4), dtype="float32"
            ),
            trainable=True,
            name="shared_distance_weights",
        )
        self.b = tf.Variable(
            tf.random_normal_initializer(seed=seed)(shape=(1, 4), dtype="float32"),
            trainable=True,
            name="shared_distance_bias",
        )

    def call(self, inputs):
        za = inputs[:, : self.dimension]
        zu = inputs[:, self.dimension : self.dimension * 2]
        zb = inputs[:, self.dimension * 2 :]
        # Keep a squared difference per coordinate, allowing the original head to weight axes.
        da = tf.math.square(tf.math.subtract(za, zu))
        db = tf.math.square(tf.math.subtract(zb, zu))
        a = tf.matmul(da, self.w)
        b = tf.matmul(db, self.w)
        return tf.exp(tf.concat([a, b], axis=1) + tf.concat([self.b, self.b], axis=1))


class DenseOriginal(tf.keras.layers.Layer):
    """One exponential score per alternative, using the same weights and bias."""

    def __init__(self, seed=101):
        super().__init__(name="dense_original")
        self.w = tf.Variable(
            tf.random_normal_initializer(seed=seed)(shape=(4, 1), dtype="float32"),
            trainable=True,
            name="shared_score_weights",
        )
        self.b = tf.Variable(
            tf.random_normal_initializer(seed=seed)(shape=(1, 1), dtype="float32"),
            trainable=True,
            name="shared_score_bias",
        )

    def call(self, inputs):
        a = tf.matmul(inputs[:, :4], self.w)
        b = tf.matmul(inputs[:, 4:], self.w)
        return tf.exp(tf.concat([a, b], axis=1) + tf.concat([self.b, self.b], axis=1))


def build_original_model(n_options, n_users, dimension=2, seed=101, embedding_seed=None):
    """Construct the original architecture; initialization and fitting are external."""
    if embedding_seed is None:
        embedding_seed = seed
    model = tf.keras.Sequential(
        [
            PoliticalCoords(n_options, n_users, dimension, embedding_seed),
            NOMOriginal(dimension, seed),
            DenseOriginal(seed),
            tf.keras.layers.Dense(1, activation="sigmoid", name="choice_probability"),
        ],
        name="dynamap_original",
    )
    model(tf.zeros((1, 3), dtype=tf.int32))
    expected_parameters = dimension * (n_options + n_users) + 4 * dimension + 12
    if model.count_params() != expected_parameters:
        raise RuntimeError("Embedding or branch variables were not tracked by Keras.")
    if len(model.trainable_variables) != 8:
        raise RuntimeError("The original architecture must have eight trainable tensors.")
    return model
