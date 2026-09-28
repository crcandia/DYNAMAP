> **Review draft: two raw inputs are pending.** `data/raw/chile_choices.csv` and `data/raw/france_participation.csv` are not included in this branch because GitHub rejects new LFS uploads from the public fork. Raw-data preparation cannot be fully rerun until both files are restored. Do not merge this draft as a complete replication release. The execution checks described for the release used the complete local package. The [data guide](data/README.md) records the expected files.

# DYNAMAP

The notebooks show their calculations, figures and tables directly. This review branch includes the fitted models and intermediate data, but notebooks 02 and 03 cannot complete Run All until the two pending raw CSV files are restored. In the complete package, each notebook can start in a fresh kernel without running the others first. Its opening section names the inputs it reads and the notebook that regenerates them.

## Start

Use Python 3.10. From this folder:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dynamap-release --display-name "DYNAMAP (release)"
python -m jupyterlab
```

Start with notebook 12 or 15 in `notebooks/`, select the **DYNAMAP (release)** kernel, and choose **Restart Kernel and Run All Cells**. Notebooks 02 and 03 require the pending raw files. The kernel registration above points to the active virtual environment, so Jupyter opened from another environment can use the same dependencies. Parameters and input paths are near the top. Figures appear inline and are saved as PDF. Tables appear as dataframes and are saved as CSV and, where used in the manuscript, LaTeX.

If you already have a Python 3.10 environment, activate it, install `requirements.txt`, and select that environment’s kernel in Jupyter. The requirements support JupyterLab 4.4 and 4.5. Restart any running kernel after installing dependencies.

Notebook 11 also needs `rsvg-convert` to export the editable architecture diagram. Install it with `brew install librsvg` on macOS or `apt install librsvg2-bin` on Debian/Ubuntu. This diagram is authored artwork. The other panels are generated from data.

## Choose a notebook

| Notebook | Contents |
|---|---|
| [01 · Age eligibility](notebooks/01_age_eligibility.ipynb) | Participant exclusions and retained samples |
| [02 · Chile](notebooks/02_chile_training.ipynb) | Raw-choice cleaning, partition and main two-dimensional model |
| [03 · France and Brazil](notebooks/03_france_brazil_training.ipynb) | Country samples, partitions and models |
| [04 · Dimensions and initializations](notebooks/04_chile_dimensions_and_initializations.ipynb) | Choice prediction across dimensions and repeated Chilean fits |
| [04b · Flexible-head fits](notebooks/04b_chile_flexible_head.ipynb) | Reload or train all five flexible-head fits used in the ablation |
| [05 · External validation](notebooks/05_external_validation.ipynb) | Ideological classification and participation groups |
| [06 · Proposal rankings](notebooks/06_proposal_rankings.ipynb) | TrueSkill rankings from retained Chilean choices |
| [09 · Figure inputs](notebooks/09_publication_inputs.ipynb) | Coordinates joined to metadata, histories and rankings |
| [10 · Sample composition](notebooks/10_sample_composition.ipynb) | Demographic and ideological sample distributions |
| [11 · Training](notebooks/11_training_history.ipynb) | Architecture diagram, coordinate snapshots and training curves |
| [12 · Spatial maps](notebooks/12_spatial_maps.ipynb) | Joint maps, participant–proposal distances and cohort/sex summaries |
| [13 · Supplementary maps](notebooks/13_supplementary_maps.ipynb) | Ranked proposals, self-placement bands, PC1 and country maps |
| [14 · Geometric stability](notebooks/14_geometric_stability.ipynb) | Positive symmetric head: distance and PC1 correlations across fits |
| [14b · Flexible-head geometry](notebooks/14b_flexible_head_geometry.ipynb) | Recompute all distance and PC1 correlations for the flexible head |
| [15 · Validation figures](notebooks/15_validation_figures.ipynb) | Classification, ROC curves, confusion matrices and panel assembly |
| [16 · Publication tables](notebooks/16_publication_tables.ipynb) | Descriptively named CSV and LaTeX tables |
| [17 · Choice geometry](notebooks/17_choice_geometry_diagnostics.ipynb) | Predictions, distance ordering and exchange symmetry |
| [18 · Choice-head ablation](notebooks/18_choice_head_ablation.ipynb) | Matched five-seed prediction and geometry comparison |

Notebook numbers are retained to keep existing references valid. The fitting notebooks for the supplied optimizer and all-cycle architecture comparisons are outside this package.

For a first look, start with 12 or 15. Notebook 05 refits the classification forests and takes about half an hour on the machine used for this release. It does not retrain DYNAMAP. Reading the raw Chilean archive in 02 also takes longer than plotting the supplied results.

## Explore or retrain

After restoring the pending inputs for 02 and 03, the complete package supports the following workflow. In 02–04 and 04b, `REFIT = False` loads the supplied weights, reevaluates choices and regenerates coordinates and plots. Change it to `True` to train using the code visible in that notebook. Full training can take hours. It overwrites the selected model's outputs, so use a copy of the folder if you want to retain the manuscript fits.

The model settings are ordinary notebook variables:

| Choice function | Settings | Choice parameters in 2D |
|---|---|---:|
| Manuscript model | `VARIANT = "intermediate"`, `HIDDEN_SIZES = [8, 8]` | 105 |
| Flexible-head comparator | `VARIANT = "original"` | 20 |
| Compact intermediate model | `VARIANT = "intermediate"`, `HIDDEN_SIZES = [3]` | 13 |

These counts exclude participant and proposal embeddings. The compact model has no supplied empirical fit. The intermediate function favors the nearer proposal and gives complementary probabilities when alternatives are swapped. Its learned response strength remains flexible.

A saved model's `model_specification.json` records how its weights must be reconstructed. It is generated metadata, not a settings file to edit. Selecting different model parameters requires a new fit. The loading cells check the architecture and ID correspondence. Changes to responses or preprocessing also require `REFIT = True`, even when the participant IDs stay the same. Loading weights does not update the map from edited choices.

## Files and dependencies

- `data/raw/` holds the available source observations and reference inputs. The two pending files are listed above. Notebooks never overwrite source inputs.
- `data/processed/` holds generated samples, splits, rankings and plotting inputs.
- `data/benchmarks/` holds supplied optimizer, architecture and ideal-point benchmark results. Their numerical inputs remain unchanged.
- `outputs/models/` and `outputs/validation/` hold fitted weights and external-validation results.
- `outputs/figures/`, `outputs/panels/`, `outputs/tables/`, `outputs/manuscript_tables/` and `outputs/diagnostics/` hold the visible analytical results.
- `assets/` contains the editable vector diagrams.
- `src/` contains only shared neural layers and the numerical TrueSkill kernel. Sample construction, validation, plotting and table assembly are visible in the notebooks.

The [data guide](data/README.md) explains file origins, key columns and which notebook generates each intermediate. If you change an upstream analysis, rerun the notebooks that use its outputs. For example, revised Chilean coordinates feed 05, 09, 14 and 17. Notebook 09 then supplies the map notebooks, while 05 supplies 15 and 16. Figure 2 assembly in 15 uses panels from 11 and 12. Publication maps target the included two-dimensional design. Structural experiments can also require changes to local figure settings or table labels. Notebook 17 can evaluate saved Chilean fits in other dimensions.

All three country maps were fitted separately. Their distances do not define a common cross-country ideological scale. The benchmark fitting scripts for W-NOMINATE and the Bayesian model are not included. Notebook 16 reproduces their tables from the author-confirmed numerical inputs.

## Choice-head ablation

The five flexible-head fits are supplied in `outputs/models/flexible/Chile_dim2_seed*`, including weights, coordinates, metrics, history and model properties. Notebook 04b reloads these fits and reevaluates every held-out choice. With `REFIT = True`, it trains them from the included first-cycle comparisons and partition using the same settings as the positive symmetric fits in 02 and 04. The two specifications are fitted separately.

Notebook 14b calculates correlations from the flexible-head coordinates using every distinct participant/proposal pair in float64, in memory-bounded blocks. It also recalculates participant PC1 rank correlations. Its outputs and the prediction metrics from 04b are in `outputs/ablation/flexible/`. The corresponding positive symmetric results are generated by 02/04 and 14. Notebook 18 reads these included outputs and assembles the comparison without refitting. After restoring the pending Chilean raw file, regenerate the full ablation by running 02 and 04, 04b, 14 and 14b, then 18. The complete package supplies the inputs needed to execute each notebook independently.

## Locate figures and tables by content

The notebook guide identifies analyses by content. Existing numerical-output filenames are stable working identifiers and need not match the supplementary numbering of a particular manuscript. Table files use descriptive names such as `participant_validation`, `geometry_across_fits` and `choice_head_ablation`. The architecture comparison is supplied as [PDF](assets/FigS1_Choice-Architectures.pdf) and [editable LaTeX/TikZ source](assets/FigS1_Choice-Architectures.tex).

## Data included for sharing

This review branch includes the processed analytical data, fitted results and source observations other than the two pending CSV files. The authors confirm that the source datasets are public. The [data guide](data/README.md) documents their provenance and variables. The two pending raw files are deliberately absent from this review commit. They remain in the complete local package and are not excluded by the local-notes ignore rule.

## GitHub and large data files

The included `.gitattributes` is prepared to track `data/raw/chile_choices.csv` and `data/raw/france_participation.csv` with Git LFS. They exceed GitHub's ordinary per-file limit. Install Git LFS and run `git lfs install` before adding this folder to the new manuscript branch. Commit the attributes with the files and check `git lfs ls-files` before pushing. The repository must have sufficient LFS storage and bandwidth. No public upload is performed by the notebooks.

Once the two files have been added to the repository through Git LFS, run `git lfs pull` after cloning. That command cannot retrieve them from this draft, where they have not been committed or uploaded. Keep the folder structure intact and open notebooks individually in JupyterLab. There is no runner or global configuration file. The `model_specification.json` files are required saved-model metadata, not an additional workflow to configure.
