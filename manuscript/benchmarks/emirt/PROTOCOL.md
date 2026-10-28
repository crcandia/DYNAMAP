# One-dimensional EM-IRT comparison

This experiment changes the latent dimension to one and compares native emIRT 0.0.15 with DYNAMAP B2 and its B1 constant-scale ablation. It uses the same training-only support masks as the two-dimensional comparison. `src/prepare.py` makes matrix and triplet views of exactly the same observed cells; missing values remain missing. It reads validation choices but no test outcomes or political labels.

The country/size ladder is Chile 300, 1,000, 5,000, 15,000 and full; France 300, 1,000 and full; Brazil 300 and full. B1 and B2 use the six-configuration grid, two tuning seeds and country tuning sizes specified in the parent protocol. Primary fits use seeds 301, 401 and 501. The neural training and stopping rules are unchanged.

EM-IRT uses the native one-dimensional binary estimator with zero starts, item prior variance 25, at most 5,000 iterations, convergence tolerance 1e-6, convtype 1, check frequency 50 and two threads. The native fitted object is saved before extracting results. Validation probabilities are checked against the native training geometric mean probability. Completion is an execution status, not a posterior-convergence verdict.

Supplementary fits use 50- and 200-iteration budgets, tolerance 1e-8, and item prior variance 6.25 at one size per country. These fits are reported separately and do not choose a primary winner. The predefined quality comparison requires an absolute validation-log-loss gap of at most 0.005. None of the 60 paired original primary comparisons met it.

The original run measured each fit serially with two CPU threads and recorded wall time, CPU time and peak memory. The portable runner keeps the two-thread setting and GNU-time measurements while using the current machine's available CPUs. It does not reproduce a hardware-independent speed ratio. Input masks, model equations, hyperparameters and seeds remain unchanged. New results are written separately from `../published/emirt_scaling/`.
