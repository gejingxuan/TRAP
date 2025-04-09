# TRAP
TRAP, an advanced deep learning framework enhanced by contrastive learning, 
which enriches the feature space beyond sequence level by incorporating structural details of pMHC, 
elucidating subtle conformational influences on binding.

# Usage
## 1. Environment
Here，you can construct the conda environment for TRAP from the requirements.txt file
In addition, we used conda pack to package the TRAP environment and uploaded it to https://zenodo.org/records/15062393

## 2. Input Data Generation
For the sake of computational efficiency, we will first generate the corresponding feature input file.
For pMHC structures in PDB format, we use `prepare.py` to extract structures from the path of AlphaFold Multimer results,
and we use `pmhc_gen.py` to extract features and generate the input data for TRAP.

```
python -u pmhc_gen.py --struc_path [your_structure_path] --epi_max [the max length of epitope] --file_path [your_pmhc_feature_file_path]
```

For TCR (cdr3 beta) sequence, we use `cdr_gen.py` to extract features and generate the input data for TRAP.

```
python -u cdr_gen.py --data_dir [your_cdr_sequence_path] --b_max [the max length of cdr3 beta] --cdr_dir [your_cdr_feature_file_path]
```
## 3. Training and Prediction
Then we use the input data generated in the last step to train and test model.

```
python -u train.py --data_dir [your_datasets_dir] --cdr_dir [your_cdr_feature_file_path] --epi_dir [your_pmhc_feature_file_path]
```
## 4. Datasets
We put the datasets used in scenarios 1 (randomly) and 2 (zero_shot) under data_split. We first split `kfold.csv` and `test.csv` according to the rules of these two scenarios, and then save the positive pair samples in `kfold.csv` (for binary classification training) as `train.csv` (for contrastive learning training).

## 5. Repository
We provide two Jupyter Notebook guides to help users get familiar with TRAP. The `repository.ipynb` guide outlines the process from data reading and feature file generation to model training and testing. The `tutorial.ipynb` guide demonstrates how to perform inference directly using the trained model parameters.


