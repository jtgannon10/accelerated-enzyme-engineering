# Accelerated enzyme engineering by machine-learning guided cell-free expression

This is the official repository for our pre-print [Accelerated enzyme engineering by machine-learning guided cell-free expression](https://www.biorxiv.org/content/10.1101/2024.07.30.605672v1). To reproduce the results in our manuscript, please use the contained Jupyter notebooks. 


## Repo contents
- `data`: Includes experimental amide synthetase fitness data from hot spot screens as well as zero-shot predictions made from ESM-1b, EVmutation, and MAESTRO. Both experimental and zero-shot predictions were used to train augmented ridge regression models. 
- `notebooks`: Two .ipynb notebooks are included that execute code to reproduce results in our manuscript. One contains our strategy for selecting an augmented model using experimental data from iterative site saturation mutagenesis campaigns. Once selecting this model, the second notebook can be used to make predictions for enzymes for the remaining set of reactions.
- `src`: Python code for the augmented ridge regression methods utilized in the paper. 
- `environment.yml`: Software dependencies for the conda environment.

