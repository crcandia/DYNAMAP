"""Closed-form, non-draw, one-against-one TrueSkill updates.

The complementary-error-function approximation below is adapted verbatim from
trueskill 0.4.5, trueskill/backends.py, copyright (c) 2012-2016 Heungsub Lee.
The BSD license is reproduced in LICENSE.trueskill. All arithmetic is float64;
Numba fastmath is disabled. This module performs no filtering, randomization,
ranking aggregation, display, or persistence. These choices are in the notebook.
"""

import math
import numpy as np
from numba import njit


@njit(fastmath=False)
def erfc_approx(x):
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    r = t * math.exp(
        -z * z
        - 1.26551223
        + t
        * (
            1.00002368
            + t
            * (
                0.37409196
                + t
                * (
                    0.09678418
                    + t
                    * (
                        -0.18628806
                        + t
                        * (
                            0.27886807
                            + t
                            * (
                                -1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
                            )
                        )
                    )
                )
            )
        )
    )
    return 2.0 - r if x < 0 else r


@njit(fastmath=False)
def rate_sequence(
    winners,
    losers,
    n_options,
    draw_margin,
    beta,
    tau,
    initial_mu,
    initial_sigma,
    record_trace=False,
):
    """Return final mu, sigma and optional per-update winner/loser states.

    Winner and loser arrays contain indices into one shared proposal universe.
    Draws are not supported because the retained Chilean input has no ties.
    Defaults are supplied explicitly by the notebook, including the global
    TrueSkill beta/tau; they are not rescaled with the proposal-count priors.
    """
    mu = np.full(n_options, initial_mu, dtype=np.float64)
    sigma = np.full(n_options, initial_sigma, dtype=np.float64)
    trace = np.empty((len(winners) if record_trace else 0, 4), dtype=np.float64)
    for i in range(len(winners)):
        winner, loser = winners[i], losers[i]
        if winner == loser:
            raise ValueError("A proposal cannot play against itself.")
        winner_variance = sigma[winner] ** 2 + tau**2
        loser_variance = sigma[loser] ** 2 + tau**2
        c_squared = winner_variance + loser_variance + 2 * beta**2
        c = math.sqrt(c_squared)
        x = (mu[winner] - mu[loser] - draw_margin) / c
        denominator = 0.5 * erfc_approx(-x / math.sqrt(2))
        v = math.exp(-(x**2) / 2) / math.sqrt(2 * math.pi) / denominator if denominator else -x
        w = v * (v + x)
        if not 0 < w < 1:
            raise FloatingPointError("TrueSkill truncation is numerically unstable.")
        mu[winner] += winner_variance / c * v
        mu[loser] -= loser_variance / c * v
        sigma[winner] = math.sqrt(winner_variance * (1 - winner_variance / c_squared * w))
        sigma[loser] = math.sqrt(loser_variance * (1 - loser_variance / c_squared * w))
        if record_trace:
            trace[i, 0] = mu[winner]
            trace[i, 1] = sigma[winner]
            trace[i, 2] = mu[loser]
            trace[i, 3] = sigma[loser]
    return mu, sigma, trace
