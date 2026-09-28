# DYNAMAP

## Manuscript package: review draft

The proposed manuscript package is in [`manuscript/`](manuscript/README.md). It contains 18 notebooks, fitted models, figures, tables, processed data, and the other source inputs. The original implementation and its documentation remain in place below.

**This review branch is incomplete and must not be merged as a complete replication release yet.** Two large source files, `manuscript/data/raw/chile_choices.csv` and `manuscript/data/raw/france_participation.csv`, are pending Git LFS upload. GitHub currently rejects new LFS objects from this fork. Raw-data preparation cannot be fully rerun from this branch until those files are restored.

The complete local package passed all 18 notebook executions with Python 3.10.13. That validation applies to the complete package, including the two pending files. This branch is for reviewing the code and organization while upstream LFS access is resolved. See the [data guide](manuscript/data/README.md) for the pending file hashes and sizes.

## Original implementation

The original files remain in their existing locations. The documentation below describes that implementation and its Python 3.9 environment.

DYNAMAP Project

This repository contains the DYNAMAP model developed in Python 3.9.16. The following diagram includes only Python files (.py), Jupyter Notebook files (.ipynb), and folders.

    └── DYNAMAP
        ├── Brazil and France
        ├── Chile First Cycle
        ├── replication_sample
        ├── data
        │   ├── labels
        ├── Library
        │   ├── ranking
        │   ├── botprediction.py
        │   ├── preferences.py
        │   └── space.py
        └── requirements.txt 
Folders:
<em><b>Brazil and France</b></em>: This folder contains the predictions of the DYNAMAP model and the analysis of results for voting processes conducted in France and Brazil.

<em><b>Chile First Cycle</b></em>: This folder contains the predictions of the DYNAMAP model and the analysis of results for the first cycle of voting conducted in Chile.

<em><b>replication_sample</b></em>: This folder provides a reduced and self-contained subset of the original data designed to facilitate replication and reproducibility of the main results. Given the large scale of the full datasets, which may pose computational and storage constraints, this sample offers a practical starting point for researchers and reviewers to validate the methodology, run the model, and reproduce key findings in a controlled setting.

<em><b>data</b></em>: Folder that stores voting information.

<em><b>Library</b></em>: Folder that contains the libraries for Preferences, Bot Prediction, and Space.

## Reproducibility

To facilitate reproducibility, we provide a reduced dataset in the `replication_sample` folder.

## Data Availability

The full datasets used in this study are large-scale and may be subject storage constraints. 

To ensure transparency and reproducibility, we provide a `replication_sample`, which allows validation of the methodology and reproduction of key results.

Instructions for accessing or reconstructing the full dataset are described in the accompanying paper.
