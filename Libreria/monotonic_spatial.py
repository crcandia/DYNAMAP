"""Experimental monotonic spatial-choice model for DYNAMAP.

This module is intentionally independent of the notebook-defined baseline.
It reproduces the baseline ``PoliticalCoords`` contract and adds a constrained
choice head; it does not change any baseline class or default.
"""

from __future__ import annotations

import math

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="DYNAMAP")
class PoliticalCoords(tf.keras.layers.Layer):
    """Serializable equivalent of the seed-0 notebook embedding layer.

    Inputs are integer indices in the order ``[option_a, user, option_b]``.
    Outputs are sigmoid-transformed coordinates in the same order.
    """

    def __init__(
        self,
        dimPoliticalSpace=2,
        nOptions=90,
        nUsers=100,
        activation="sigmoid",
        initializer_stddev=0.05,
        initializer_seed=101,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(dimPoliticalSpace, int) or dimPoliticalSpace <= 0:
            raise ValueError("`dimPoliticalSpace` must be a positive integer.")
        if not isinstance(nOptions, int) or nOptions <= 0:
            raise ValueError("`nOptions` must be a positive integer.")
        if not isinstance(nUsers, int) or nUsers <= 0:
            raise ValueError("`nUsers` must be a positive integer.")
        self.dimPoliticalSpace = dimPoliticalSpace
        self.nOptions = nOptions
        self.nUsers = nUsers
        self.activation = activation
        self.initializer_stddev = float(initializer_stddev)
        self.initializer_seed = initializer_seed

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 2:
            raise ValueError("PoliticalCoords expects rank-2 input.")
        if input_shape[-1] is not None and int(input_shape[-1]) != 3:
            raise ValueError("PoliticalCoords expects [option_a, user, option_b].")
        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0,
            stddev=self.initializer_stddev,
            seed=self.initializer_seed,
        )
        self.wp = self.add_weight(
            name="option_coordinates",
            shape=(self.nOptions, self.dimPoliticalSpace),
            initializer=initializer,
            trainable=True,
        )
        self.wu = self.add_weight(
            name="user_coordinates",
            shape=(self.nUsers, self.dimPoliticalSpace),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=self.initializer_stddev,
                seed=self.initializer_seed,
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.int32)
        tf.debugging.assert_equal(
            tf.shape(inputs)[-1], 3, message="Expected [option_a, user, option_b]."
        )
        z_a = tf.gather(self.wp, inputs[:, 0])
        z_user = tf.gather(self.wu, inputs[:, 1])
        z_b = tf.gather(self.wp, inputs[:, 2])
        return tf.math.sigmoid(tf.concat([z_a, z_user, z_b], axis=-1))

    def returnOptionsEmbedding(self, withActivation=True):  # noqa: N802
        del withActivation
        return tf.math.sigmoid(self.wp)

    def returnUsersEmbedding(self, withActivation=True):  # noqa: N802
        del withActivation
        return tf.math.sigmoid(self.wu)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dimPoliticalSpace": self.dimPoliticalSpace,
                "nOptions": self.nOptions,
                "nUsers": self.nUsers,
                "activation": self.activation,
                "initializer_stddev": self.initializer_stddev,
                "initializer_seed": self.initializer_seed,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="DYNAMAP")
class MonotonicSpatialChoice(tf.keras.layers.Layer):
    """Return ``sigmoid(tau * (d_b - d_a))`` for ``[A, user, B]``."""

    def __init__(self, dimension: int, minimum_tau: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("`dimension` must be a strictly positive integer.")
        if minimum_tau <= 0:
            raise ValueError("`minimum_tau` must be strictly positive.")
        self.dimension = dimension
        self.minimum_tau = float(minimum_tau)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        expected_features = 3 * self.dimension
        if input_shape.rank != 2:
            raise ValueError(
                "MonotonicSpatialChoice expects rank-2 input with shape "
                f"(batch_size, {expected_features}); received {input_shape}."
            )
        observed_features = input_shape[-1]
        if observed_features is not None and int(observed_features) != expected_features:
            raise ValueError(
                f"Final input dimension must be {expected_features}; "
                f"received {observed_features}."
            )
        # The initial tau is approximately 1 + minimum_tau for every valid epsilon.
        initial_raw_tau = math.log(math.expm1(1.0))
        self.raw_tau = self.add_weight(
            name="raw_tau",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_raw_tau),
            trainable=True,
        )
        super().build(input_shape)

    @property
    def tau(self):
        return tf.nn.softplus(self.raw_tau) + tf.cast(
            self.minimum_tau, self.raw_tau.dtype
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, self.compute_dtype)
        expected_features = 3 * self.dimension
        tf.debugging.assert_rank(
            inputs, 2, message="MonotonicSpatialChoice expects rank-2 input."
        )
        tf.debugging.assert_equal(
            tf.shape(inputs)[-1],
            expected_features,
            message="Incompatible coordinate tensor.",
        )
        z_a, z_user, z_b = tf.split(inputs, 3, axis=-1)
        distance_a = tf.reduce_sum(
            tf.square(z_a - z_user), axis=-1, keepdims=True
        )
        distance_b = tf.reduce_sum(
            tf.square(z_b - z_user), axis=-1, keepdims=True
        )
        return tf.math.sigmoid(self.tau * (distance_b - distance_a))

    def get_config(self):
        config = super().get_config()
        config.update(
            {"dimension": self.dimension, "minimum_tau": self.minimum_tau}
        )
        return config


def build_monotonic_spatial_model(
    dimPoliticalSpace=2,
    nOptions=90,
    nUsers=100,
    initializer_stddev=0.05,
    initializer_seed=101,
    minimum_tau=1e-6,
    optimizer=None,
    compile_model=True,
    political_coords_layer=None,
):
    """Build only the experimental monotonic spatial model.

    An already-created compatible ``PoliticalCoords`` layer may be supplied so
    notebook experiments can use the exact baseline-defined class instance.
    """
    inputs = tf.keras.Input(shape=(3,), dtype=tf.int32, name="pairwise_input")
    if political_coords_layer is None:
        political_coords_layer = PoliticalCoords(
            nOptions=nOptions,
            nUsers=nUsers,
            dimPoliticalSpace=dimPoliticalSpace,
            activation="sigmoid",
            initializer_stddev=initializer_stddev,
            initializer_seed=initializer_seed,
            name="political_coords",
        )
    coordinates = political_coords_layer(inputs)
    probabilities = MonotonicSpatialChoice(
        dimension=dimPoliticalSpace,
        minimum_tau=minimum_tau,
        name="monotonic_spatial_choice",
    )(coordinates)
    model = tf.keras.Model(
        inputs=inputs,
        outputs=probabilities,
        name="dynamap_monotonic_spatial",
    )
    if compile_model:
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adamax(learning_rate=0.01)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    return model
