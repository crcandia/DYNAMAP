import json
from pathlib import Path
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

sys.path.insert(0, str(Path("Libreria").resolve()))
from monotonic_spatial import (  # noqa: E402
    MonotonicSpatialChoice,
    build_monotonic_spatial_model,
)


def _coords(z_a, z_user, z_b):
    return tf.constant([z_a + z_user + z_b], dtype=tf.float32)


def test_a_closer_than_b():
    assert float(MonotonicSpatialChoice(2)(_coords([0, 0], [0, 0], [1, 0]))) > 0.5


def test_b_closer_than_a():
    assert float(MonotonicSpatialChoice(2)(_coords([1, 0], [0, 0], [0, 0]))) < 0.5


def test_equal_distance():
    probability = MonotonicSpatialChoice(2)(_coords([-1, 0], [0, 0], [1, 0]))
    np.testing.assert_allclose(probability.numpy(), [[0.5]], atol=0.0, rtol=0.0)


def test_swap_symmetry():
    layer = MonotonicSpatialChoice(2)
    p_ab = layer(_coords([0.1, 0.2], [0.3, 0.4], [0.8, 0.9]))
    p_ba = layer(_coords([0.8, 0.9], [0.3, 0.4], [0.1, 0.2]))
    np.testing.assert_allclose(p_ab + p_ba, [[1.0]], atol=1e-6, rtol=0.0)


def test_random_monotonicity_and_swap_symmetry():
    tf.random.set_seed(1234)
    dimension = 2
    z_a = tf.random.uniform((1000, dimension))
    z_user = tf.random.uniform((1000, dimension))
    z_b = tf.random.uniform((1000, dimension))
    layer = MonotonicSpatialChoice(dimension)
    p_ab = layer(tf.concat([z_a, z_user, z_b], axis=-1))
    p_ba = layer(tf.concat([z_b, z_user, z_a], axis=-1))
    d_a = tf.reduce_sum(tf.square(z_user - z_a), axis=-1, keepdims=True)
    d_b = tf.reduce_sum(tf.square(z_user - z_b), axis=-1, keepdims=True)
    assert bool(tf.reduce_all(tf.boolean_mask(p_ab, d_a < d_b) > 0.5))
    assert bool(tf.reduce_all(tf.boolean_mask(p_ab, d_a > d_b) < 0.5))
    np.testing.assert_allclose(p_ab + p_ba, np.ones_like(p_ab), atol=1e-6, rtol=0.0)


def test_tau_positive_before_and_after_optimization():
    layer = MonotonicSpatialChoice(2)
    values = _coords([0, 0], [0, 0], [1, 1])
    _ = layer(values)
    assert float(layer.tau) > 0
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(layer(values))
    gradient = tape.gradient(loss, layer.raw_tau)
    tf.keras.optimizers.SGD(0.1).apply_gradients([(gradient, layer.raw_tau)])
    assert float(layer.tau) > 0


def test_output_shape_and_finite_gradients():
    layer = MonotonicSpatialChoice(2)
    values = tf.Variable(tf.random.uniform((8, 6), seed=7))
    with tf.GradientTape() as tape:
        probabilities = layer(values)
        loss = tf.reduce_sum(probabilities)
    coordinate_gradient, tau_gradient = tape.gradient(loss, [values, layer.raw_tau])
    assert probabilities.shape == (8, 1)
    assert coordinate_gradient is not None and bool(tf.reduce_all(tf.math.is_finite(coordinate_gradient)))
    assert tau_gradient is not None and bool(tf.math.is_finite(tau_gradient))


def test_serialization_round_trip(tmp_path):
    inputs = tf.keras.Input(shape=(6,))
    model = tf.keras.Model(inputs, MonotonicSpatialChoice(2)(inputs))
    values = tf.random.uniform((4, 6), seed=9)
    expected = model(values).numpy()
    path = tmp_path / "monotonic_spatial.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(restored(values).numpy(), expected, atol=1e-7, rtol=0.0)


def test_complete_builder_serialization_round_trip(tmp_path):
    model = build_monotonic_spatial_model(nOptions=5, nUsers=4)
    values = tf.constant([[0, 0, 1], [2, 1, 3], [4, 2, 0]])
    expected = model(values).numpy()
    path = tmp_path / "dynamap_monotonic_spatial.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(restored(values).numpy(), expected, atol=1e-7, rtol=0.0)


def test_target_convention_matches_logit_sign():
    # The seed-0 formatData sets Y=1 exactly when selected == option_a.
    notebook = Path("Chile Primer Ciclo/Entrenamiento Usuarios y propuestas - semilla 0.ipynb")
    sources = "\n".join("".join(c["source"]) for c in json.loads(notebook.read_text())["cells"])
    assert "df.loc[df['selected'] == df['option_a'], 'Y'] = 1" in sources
    assert float(MonotonicSpatialChoice(1)(_coords([0], [0], [1]))) > 0.5


def test_baseline_notebook_still_defines_and_builds_original_model():
    notebook = Path("Chile Primer Ciclo/Entrenamiento Usuarios y propuestas - semilla 0.ipynb")
    cells = json.loads(notebook.read_text())["cells"]
    architecture = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and "class PoliticalCoords" in "".join(cell["source"])
    )
    namespace = {
        "tf": tf,
        "Layer": tf.keras.layers.Layer,
        "Dense": tf.keras.layers.Dense,
        "Sequential": tf.keras.Sequential,
    }
    exec(architecture, namespace)
    baseline = namespace["get_model_original"](dimPoliticalSpace=2, nOptions=5, nUsers=4)
    output = baseline(tf.constant([[0, 0, 1], [2, 1, 3]], dtype=tf.int32))
    assert output.shape == (2, 1)


def test_experimental_builder_forward_and_train_step():
    model = build_monotonic_spatial_model(nOptions=5, nUsers=4)
    x = tf.constant([[0, 0, 1], [2, 1, 3], [4, 2, 0], [1, 3, 2]])
    y = tf.constant([1.0, 0.0, 1.0, 0.0])
    before = model(x)
    loss = model.train_on_batch(x, y)
    assert before.shape == (4, 1)
    assert bool(np.all(np.isfinite(np.asarray(loss))))
    choice = model.get_layer("monotonic_spatial_choice")
    assert float(choice.tau) > 0
