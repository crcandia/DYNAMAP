# Inputs for notebook 19

These are included derived extracts for the fixed within-self-placement analysis, not new raw observations. The raw Chilean observations remain in `../../raw`. Every file needed by notebook 19 is in this directory. It reads no pilot folder and requires no previously generated table or figure.

| File | Contents and provenance |
|---|---|
| `participants.csv` | 3,695 participants reporting exactly 3, with `uuid`, original embedding index `u`, and the original discovery (1,825) or confirmation (1,870) assignment. These participant assignments are fixed analytical inputs. |
| `{101,201,301,401,501}/coordinates.csv` | The same participants' coordinates extracted from the five existing positive symmetric Chilean fits. The complete fits are supplied in `../../../outputs/models/Chile_dim2_seed*`. |
| `{101,201,301,401,501}/proposals.csv` | All 90 proposal IDs and coordinates in each corresponding fit. |
| `sector_thresholds.csv` | Original discovery distance terciles relative to proposal 37 for each fit. Notebook 19 recomputes and checks them. |
| `confirmation_choices.parquet` | 13,908 reserved comparisons for the 1,870 confirmation participants. Derived from the original Chilean test partition. Contains no training choices. `option_a`, `option_b`, `selected` are proposal IDs, `a` and `b` index the supplied proposal order, `u` indexes participants, and `y=1` denotes choosing A. |
| `proposal_texts.tsv` | Original available proposal wording and topic labels, copied from the supplied Chilean label table. The `name` field is the full available name, not an extended legal description. |
| `figure_labels.csv` | Descriptive English translations for the six frozen proposals. |
| `age_eligibility.csv` | Existing inclusion decisions for these participants. Known ages below 19 were excluded and unknown ages retained. Notebook 19 imposes no new eligibility rule. |
| `selection_lock.json` | Original timestamp, selected IDs and hashes. Read-only analytical provenance, not a parameter configuration. |
| `selection_protocol.md` | Verbatim historical selection protocol from 14 October 2026. Its operational instruction not to edit the release applied to that original task. The subsequent release integration was separately authorized. |

The extracts were copied unchanged from the completed `dynamap_map_value_20261014` analysis. That source is historical provenance only and is never accessed by this notebook. The original participant split and case choice are retained, not rerandomized. The notebook regenerates rankings, proposal selection, observed contrasts, participant bootstrap intervals, stability summaries and the figure from these included inputs. No weights or complete training database are required.

Changing the upstream maps does not silently update this frozen extract. A new analysis would require rebuilding its inputs and explicitly reconsidering the locked case. The secondary historical evaluation and pointwise interval limitations remain documented in the notebook.
