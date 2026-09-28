"""Selectable DYNAMAP heads, without changing the historical implementation.

The intermediate head guarantees A/B exchange symmetry and preference for the
Euclidean-closer option. It does not impose monotone probability derivatives
with respect to either distance, or a separable utility for each proposal.
"""

import json
import math
from numbers import Integral

import tensorflow as tf

from model import PoliticalCoords, build_original_model


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@tf.keras.utils.register_keras_serializable(package="DYNAMAP")
class SymmetricPositiveChoice(tf.keras.layers.Layer):
    """sigmoid((sB-sA)*g(sA,sB)), with positive symmetric learned g.

    Input columns are concatenated coordinates [A, participant, B]. Distances
    sA and sB are squared Euclidean distances. The shared MLP sees only
    [(sA+sB)/(2*K), ((sA-sB)/K)**2], so reversing A/B cannot change g.
    Guarantees hold in exact arithmetic, subject to numerical precision.
    """

    def __init__(self, dimension, hidden_sizes=(8, 8), seed=101, minimum_scale=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dimension = _integer(dimension, "dimension")
        self.seed = _integer(seed, "seed", minimum=0)
        if not isinstance(hidden_sizes, (list, tuple)):
            raise ValueError("hidden_sizes must be a list or tuple of positive integers")
        self.hidden_sizes = tuple(_integer(size, "hidden size") for size in hidden_sizes)
        self.minimum_scale = float(minimum_scale)
        if not math.isfinite(self.minimum_scale) or not 0 < self.minimum_scale < 1:
            raise ValueError("minimum_scale must be finite and between 0 and 1")
        self.input_spec = tf.keras.layers.InputSpec(ndim=2, axes={-1: 3 * self.dimension})
        self.hidden_layers = [
            tf.keras.layers.Dense(
                size,
                activation="tanh",
                name=f"scale_hidden_{index+1}",
                kernel_initializer=tf.keras.initializers.GlorotUniform(
                    seed=self.seed + index + 1
                ),
                bias_initializer="zeros",
                dtype=self.dtype_policy,
            )
            for index, size in enumerate(self.hidden_sizes)
        ]
        self.scale_output = tf.keras.layers.Dense(
            1,
            name="raw_scale",
            dtype=self.dtype_policy,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=1e-3,
                seed=self.seed + len(self.hidden_sizes) + 1,
            ),
            bias_initializer=tf.keras.initializers.Constant(
                math.log(math.expm1(1 - self.minimum_scale)),
            ),
        )

    def build(self, input_shape):
        shape = tf.TensorShape(input_shape)
        if shape.rank != 2 or (shape[-1] is not None and shape[-1] != 3 * self.dimension):
            raise ValueError("Expected rank-2 concatenated coordinates [A, participant, B]")
        feature_shape = tf.TensorShape([shape[0], 2])
        for layer in self.hidden_layers:
            layer.build(feature_shape)
            feature_shape = tf.TensorShape([shape[0], layer.units])
        self.scale_output.build(feature_shape)
        super().build(input_shape)

    def distance_scale(self, s_a, s_b):
        """Evaluate g on a built layer; inputs have final dimension one."""
        s_a = tf.cast(s_a, self.compute_dtype)
        s_b = tf.cast(s_b, self.compute_dtype)
        # Both features are unchanged by exchanging s_a and s_b.
        # The first measures overall remoteness, the second how unequal the distances are.
        features = tf.concat(
            [
                (s_a + s_b) / (2 * self.dimension),
                tf.square((s_a - s_b) / self.dimension),
            ],
            axis=-1,
        )
        for layer in self.hidden_layers:
            features = layer(features)
        # softplus makes the multiplier positive without limiting its upper value.
        return tf.nn.softplus(self.scale_output(features)) + self.minimum_scale

    def call(self, inputs):
        inputs = tf.cast(inputs, self.compute_dtype)
        tf.debugging.assert_rank(inputs, 2)
        tf.debugging.assert_equal(tf.shape(inputs)[-1], 3 * self.dimension)
        a, participant, b = tf.split(inputs, 3, axis=-1)
        s_a = tf.reduce_sum(tf.square(a - participant), axis=-1, keepdims=True)
        s_b = tf.reduce_sum(tf.square(b - participant), axis=-1, keepdims=True)
        # A positive scale preserves the sign: the closer proposal has probability above 0.5.
        return tf.math.sigmoid((s_b - s_a) * self.distance_scale(s_a, s_b))

    def get_config(self):
        config = super().get_config()
        config.update(
            dimension=self.dimension,
            hidden_sizes=list(self.hidden_sizes),
            seed=self.seed,
            minimum_scale=self.minimum_scale,
        )
        return config


def build_choice_model(
    n_options,
    n_users,
    dimension=2,
    seed=101,
    embedding_seed=None,
    variant="intermediate",
    hidden_sizes=(8, 8),
    minimum_scale=1e-6,
):
    """Build an uncompiled model. Rebuild from the saved specification before loading H5 weights.

    `original` dispatches to the unchanged historical constructor. Only the
    intermediate mode uses hidden_sizes and minimum_scale. The configuration
    is attached as plain JSON text, avoiding TensorFlow tracking wrappers.
    Both modes retain the embedding as model.layers[0].
    """
    n_options = _integer(n_options, "n_options")
    n_users = _integer(n_users, "n_users")
    dimension = _integer(dimension, "dimension")
    seed = _integer(seed, "seed", minimum=0)
    embedding_seed = (
        seed
        if embedding_seed is None
        else _integer(
            embedding_seed,
            "embedding_seed",
            minimum=0,
        )
    )
    configuration = dict(
        n_options=n_options,
        n_users=n_users,
        dimension=dimension,
        seed=seed,
        embedding_seed=embedding_seed,
        variant=variant,
    )
    if variant == "original":
        model = build_original_model(n_options, n_users, dimension, seed, embedding_seed)
    elif variant == "intermediate":
        head = SymmetricPositiveChoice(
            dimension,
            hidden_sizes,
            seed,
            minimum_scale,
            name="symmetric_positive_choice",
        )
        model = tf.keras.Sequential(
            [
                PoliticalCoords(n_options, n_users, dimension, embedding_seed),
                head,
            ],
            name="dynamap_intermediate",
        )
        model(tf.zeros((1, 3), dtype=tf.int32))
        configuration.update(
            hidden_sizes=list(head.hidden_sizes), minimum_scale=head.minimum_scale
        )
    else:
        raise ValueError("variant must be 'original' or 'intermediate'")
    model.choice_configuration_json = json.dumps(configuration, sort_keys=True)
    return model
