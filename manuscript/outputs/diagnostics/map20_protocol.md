# Frozen design for notebook 20

Design fixed before examining the global distance-agreement relation or any self-placement-7 confirmation contrast. This is a secondary analysis of historically consulted data, not preregistration of a new experiment.

## Inputs and population

Use the release's retained Chilean observations, original train/test assignments, age decisions and the existing positive-symmetric two-dimensional fits 101, 201, 301, 401 and 501. No model fitting. Retain participants with one unambiguous valid integer self-placement from 0 through 10, including 5. Preserve the existing UUID-based discovery/confirmation rule from notebook 19's source analysis. Preserve the exact case-3 population, anchor, thresholds and selected proposals from notebook 19.

## Global analysis

Use test choices from confirmation participants. Canonize the alternative pair and orientation. Within each exact self-placement and unordered alternative pair, form each unordered pair of different participants once. Agreement equals choosing the same alternative. Repeated comparisons by the same two people on different alternative pairs are retained and remain dependent. Check that each participant has at most one record per unordered alternative pair before creating dyads.

Primary statistic: a linear probability association of agreement with Euclidean participant distance, including fixed effects for exact self-placement by alternative pair. Normalize distance by the discovery dyad 90th-minus-10th percentile range. The negative slope, multiplied by 100, is the agreement advantage for a distance change from the discovery 90th to 10th percentile. This is an association and a linear summary, not a causal effect. The estimand weights eligible response dyads equally, not participants equally.

Visualization: five distance bins, with boundaries determined from discovery dyad distances in each fit, and a separate fixed-effect categorical distance model. Residualize both outcomes and regressors within strata. Standardize the fitted category means to the complete eligible confirmation-dyad composition. Do not discard strata solely because they lack all five bins. Report how many strata and dyads actually provide within-stratum distance variation.

Uncertainty: 1,000 participant bootstrap replicates, resampling participants within exact self-placement. Every person receives one weight shared across all of their records. A dyad receives the product of both endpoints' weights. Reestimate fixed effects and denominators in every replicate. Intervals are pointwise and conditional on the map, frozen bins and the observed set of alternatives.

Reference: 499 position permutations within exact self-placement among confirmation participants. Each person is assigned one other person's whole coordinate vector, reused in every dyad. Use the same frozen distance normalizer, bins and fixed-effect estimator. This reference tests the observed correspondence between positions and decisions under within-label exchangeability. It is not a randomized experiment or causal test. Compute the one-sided Monte Carlo tail for a more negative primary slope, with the plus-one correction.

Use seed 101 as primary. Repeat the same analysis in the other four supplied maps as sensitivity, not independent replication. Random seed 20261015. Do not retune bins, weights, samples or the primary statistic after inspecting results.

## Case 7

Use only self-placement-7 discovery participants for selection. For every proposal, calculate the IQR of its within-person rank among all 90 alternatives in each of the five maps. An anchor is eligible if its seed-101 distance terciles have at least 30 distinct discovery test participants per extreme and at least 10 opposing proposals seen in both extremes. Eligibility uses exposure only, never which alternative was chosen. Select the eligible proposal with the largest median IQR across fits, breaking ties by proposal ID. If none qualifies, report insufficient support without relaxing the rule.

Define Near, Middle and Far sectors by discovery distance terciles to this one anchor. Select at most six proposals by the same rule as notebook 19: the anchor, up to two distinctive proposals per extreme with mean within-person rank at most 30 and advantage at least 10 ranks, and one shared-priority proposal with both mean ranks at most 30 and absolute gap at most 5. Use notebook 19's deterministic sorting and retain every selected proposal. Freeze the anchor, thresholds and selected IDs before examining confirmation outcomes.

Evaluate both cases with the notebook-19 opponent fixed-effect contrast and 2,000 participant bootstrap replicates. Preserve seed 20261014 for exact case-3 reproduction, and use seed 20261015 for case 7. Report total and common-opponent people and choices. Retain uncertain, contradictory and low-support estimates. Rankings and membership stability are checked across all five fits. Newly estimated behavioral contrasts use the fixed principal-map sectors.

## Outputs and stopping rule

One executable notebook 20, one global figure and one case figure for each self-placement, together with the generating tables and provenance. Native inline outputs, PDF figures and structured LaTeX summary tables. No manuscript editing, new benchmarks, global clustering, retraining or replacement of an unfavorable case. Report what the data support and stop.
