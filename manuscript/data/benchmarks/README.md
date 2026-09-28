# Comparison inputs used in the manuscript

These files are the fixed numerical inputs for comparisons that remain in the paper. Notebook 16 reads them, displays the resulting dataframes and exports Tables S1, S2, S5, S6, S7 and S10.

- `optimizer_comparison.csv` and `learning_rate_comparison.csv` contain the predecessor-model experiments for S1 and S2.
- `architecture_results.csv` contains the six all-cycle architectures for S10. `architecture_parameters/` supplies their measured tensor counts.
- `benchmark_runtime.csv`, `benchmark_accuracy.csv` and `benchmark_coverage.csv` preserve the author-confirmed values for S5–S7. The CSVs were transcribed from the author-confirmed manuscript tables without changing values. No new benchmark estimates were calculated during that extraction.

These comparisons do not describe the positive symmetric head used in the main results. No earlier fitted model, coordinate table or figure is required to render them. Fitting code for the author-supplied W-NOMINATE and Bayesian comparisons is outside this package.
