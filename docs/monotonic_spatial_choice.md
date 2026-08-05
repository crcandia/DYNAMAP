# Experimental monotonic spatial choice

## Status and baseline

This is an additive experiment. It does not replace the seed-0 baseline or
change any default. The baseline remains defined in
`Chile Primer Ciclo/Entrenamiento Usuarios y propuestas - semilla 0.ipynb` as:

```text
[option_a, user, option_b]
  -> PoliticalCoords
  -> NOM_original
  -> DENSE_original
  -> Dense(1, sigmoid)
```

`PoliticalCoords` learns shared proposal and user coordinates, applies a
sigmoid to them, and concatenates them as `[z_a, z_user, z_b]`. With latent
dimension `K`, its output shape is `(batch_size, 3K)`. The flexible baseline
head can fit pairwise choices well, but unrestricted weights and biases do not
mathematically force predicted preference to follow Euclidean proximity.

The real target convention is set by `formatData`: `Y=1` exactly when
`selected == option_a`; after draws are removed, `Y=0` means B was selected.

## Experimental architecture

The experimental path is:

```text
[option_a, user, option_b]
  -> PoliticalCoords
  -> MonotonicSpatialChoice
  -> predicted P(option_a selected)
```

For shared coordinates `z_a`, `z_u`, and `z_b`, it computes squared Euclidean
distances

```text
d_a = sum_k (z_a[k] - z_u[k])^2
d_b = sum_k (z_b[k] - z_u[k])^2
```

and the probability

```text
P(A | u, A, B) = sigmoid[tau * (d_b - d_a)]
tau = softplus(raw_tau) + epsilon > 0
```

There is one global `tau`, initialized near one. It controls how sharply a
given distance difference changes probability. There is no bias, positional
weight, proposal intercept, Dense layer, or MLP after the distances.

Consequently, `d_a < d_b` implies `P(A) > 0.5`, equal distances imply `0.5`,
and swapping A and B negates the logit. Thus the two first-position
probabilities sum to one up to floating-point precision. Gradients still flow
to `raw_tau` and to the shared proposal and user embeddings.

## Files and use

- `Libreria/monotonic_spatial.py` contains the serializable experimental
  embedding contract, `MonotonicSpatialChoice`, and
  `build_monotonic_spatial_model`.
- `tests/test_monotonic_spatial_choice.py` contains fast structural,
  serialization, gradient, target-convention, and baseline-integrity tests.
- `scripts/smoke_monotonic_spatial.py` performs a synthetic forward pass and
  training step and checks swap symmetry.
- `Chile Primer Ciclo/Entrenamiento Usuarios y propuestas - semilla 0 - monotonic spatial.ipynb`
  prepares the seed-0 comparison and writes only experimental artifacts under
  `results/monotonic_spatial/`.

Build the experimental model from the repository root:

```python
import sys
sys.path.insert(0, "Libreria")
from monotonic_spatial import build_monotonic_spatial_model

model = build_monotonic_spatial_model(
    nOptions=n_options,
    nUsers=n_users,
    dimPoliticalSpace=2,
)
```

The unchanged baseline is built in its original notebook with
`get_model_original(...)`.

Run validation with a TensorFlow 2.11-compatible environment:

```bash
python -m pytest -q
python scripts/smoke_monotonic_spatial.py
```

Run the comparison by opening the experimental notebook from `Chile Primer
Ciclo/` and executing all cells. It preserves the baseline's final per-user
10% test set and divides the original training portion into shared train and
validation sets (81/9/10 overall). Both arms use the same seed, optimizer,
learning rate, batch size, epochs, callbacks, and input/target convention. The
original notebook uses its test set for validation and has no separate
validation split; it remains unchanged. The live comparison table is displayed before it is saved to
`results/monotonic_spatial/metrics_monotonic_spatial.csv`.

## Guarantees and limits

The predicted preferred alternative is always the closer one in the learned
coordinates. This does not imply every observed choice is perfectly spatial,
does not guarantee perfect accuracy or identification of an absolute
ideological orientation, and does not select the best latent dimension. The
shared coordinate system remains jointly transformable; users and proposals
must never be rotated or reflected separately. The experiment remains a model
for review and empirical comparison, not a baseline replacement.
