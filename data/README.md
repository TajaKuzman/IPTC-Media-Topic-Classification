# Data

This folder contains all of the data used in the experiments.

The folder structure is as follows:

- **raw.** The original, immutable data dump.
- **interim.** Intermediate data that has been transformed.
- **processed.** The final, canonical data sets for modeling.
- **external.** Data from third-party sources.

The folder structure has to be created. Since the data can be provided by
DVC ([see home page](../README.md)) and is thus ignored in git, the described
folder structure has to be created by the user.
