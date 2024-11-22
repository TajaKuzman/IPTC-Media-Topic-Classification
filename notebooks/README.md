# Notebooks

The folder contains the Jupyter notebooks.

## Instructions

### Naming Conventions

Since notebooks are a challenging object for source control, we will not collaborate directly with others on Jupyter notebooks. There are two steps to use to collaborate effectively:

1. Follow a naming convention that shows the owner and the order the analysis was done. Use the following naming format: `<step>-<ghuser>-<description>.ipynb`. Example: `01-username-raw-data-analysis.ipynb`.

2. Refactor the good parts. To avoid code duplication (i.e. having the same code in multiple parts), put the code in the appropriate `/src` folder, and import it into the notebooks.
