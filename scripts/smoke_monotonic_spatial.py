"""Fast synthetic smoke test for the experimental monotonic spatial model."""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Libreria"))
from monotonic_spatial import build_monotonic_spatial_model


def main():
    tf.random.set_seed(101)
    rng = np.random.default_rng(101)
    n_options, n_users, n_rows = 12, 8, 256
    x = np.column_stack(
        [
            rng.integers(n_options, size=n_rows),
            rng.integers(n_users, size=n_rows),
            rng.integers(n_options, size=n_rows),
        ]
    ).astype("int32")
    y = rng.integers(2, size=n_rows).astype("float32")
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
    baseline = namespace["get_model_original"](
        dimPoliticalSpace=2, nOptions=n_options, nUsers=n_users
    )
    baseline_output = baseline(x[:8])

    model = build_monotonic_spatial_model(nOptions=n_options, nUsers=n_users)
    initial = model(x[:8], training=False)
    coordinate_layer = model.get_layer("political_coords")
    with tf.GradientTape() as tape:
        probabilities = model(x[:8], training=True)
        gradient_loss = tf.reduce_mean(probabilities)
    gradients = tape.gradient(
        gradient_loss,
        [coordinate_layer.wp, coordinate_layer.wu, model.get_layer("monotonic_spatial_choice").raw_tau],
    )
    history = model.fit(x, y, batch_size=64, epochs=1, verbose=0)
    swapped = x[:, [2, 1, 0]]
    p_ab = model.predict(x, verbose=0)
    p_ba = model.predict(swapped, verbose=0)
    swap_error = np.abs(p_ab + p_ba - 1.0)
    choice = model.get_layer("monotonic_spatial_choice")
    assert initial.shape == (8, 1)
    assert baseline_output.shape == (8, 1)
    assert all(gradient is not None for gradient in gradients)
    assert all(
        np.all(np.isfinite(tf.convert_to_tensor(gradient).numpy()))
        for gradient in gradients
    )
    assert np.isfinite(history.history["loss"][-1])
    assert float(choice.tau.numpy()) > 0
    assert float(swap_error.max()) < 1e-6
    print(f"loss={history.history['loss'][-1]:.8f}")
    print(f"tau={float(choice.tau.numpy()):.8f}")
    print(f"swap_mean_error={float(swap_error.mean()):.3e}")
    print(f"swap_max_error={float(swap_error.max()):.3e}")


if __name__ == "__main__":
    main()
