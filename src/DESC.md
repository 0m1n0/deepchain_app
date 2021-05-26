# Description
This app is a base application that compute the probability if the proteins belong to human.
Scores are computed using SGDClassifier (linear classifier with Stochastic Gradient Descent training),
    provided by sklearn.linear_model. Two features are used:
        - Embeddings: given by DeepChain using BioTransformers
        - One-hot encoding: categorical variables (amino acid) are represented as binary vectors using OneHotEncoder.

# Tags

## libraries
- numpy
- scipy
- sklearn
- biodatasets
- biotransformers
- deepchain.components
- torch
- joblib
- loguru
- tqdm
- statistics
- matplotlib

## tasks
- Classification
- SGD

## embeddings
- protbert(cls)

## datasets

