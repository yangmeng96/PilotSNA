# Segmented Neural Network-Driven Risk Factor Aggregation

This is the repo of Xueting Ding, Yang Meng, Liner Xiang's work.

The pilot results and figrues are stored in folder 'data/' and 'figs/'.
 
## To run the experiments:
 
1. Run "raw_data_preprocessing.sas" and output a cleaned research file.

2. Put the raw data (a ".sas7bdat" file) into /data path

3. Run "run_experiments.ipynb", uncomment "# preprocessing(filename)" to preprocess the data

4. Run "roc_graph.ipynb"  to draw the ROC and PR plots.

## baseline.py

Given the mapping of "a big disease group: a few codes", we pick one covariate/ code from the disease group using maximum aggregation. 

## pca.py

Given the mapping of "a big disease group: a few codes", we apply PCA to this group and pick the top components in the downstream tasks.

## pca_all.py

We apply PCA to all the disease-related covariates and pick the top components in the downstream tasks.

## ae.py

Given the mapping of "a big disease group: a few codes", we apply a two layer autoencoder, with 1-dimensional hidden variable z for each category-based encoder, then combine all the z as the features.

## nn.py

Given the mapping of "a big disease group: a few codes", we apply a two layer neural network to this group and output a feature of this group, combine all the features for the downstream tasks.

## lr.py

Build Logistic regression model.

## rf.py

Build Random Forest classifier.

## roc.py

Draw the ROC plot of predictions made under test data.

## pr.py

Draw the PR plot of predictions made under test data.
