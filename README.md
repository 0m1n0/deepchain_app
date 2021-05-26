# Description

This project aims to evaluate protein sequences if they belong to humans or pathogens.
It is a collaborative framework provided by DeepChain apps. The main [deepchain-apps](https://pypi.org/project/deepchain-apps/) package can be found on pypi.
To leverage the apps capability, take a look at the [bio-transformers](https://pypi.org/project/bio-transformers/) and [bio-datasets](https://pypi.org/project/bio-datasets) package.


# Usage
Linear classifiers with SGD (stochastic gradient descent) training,  [sklearn.linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html), is applied on two types of features:
 -  Probert embeddings: given by [deepchain-apps](https://pypi.org/project/deepchain-apps/) using [bio-transformers](https://pypi.org/project/bio-transformers/)
 -  One-hot encoding: categorical variables (amino acid) are represented as binary vectors using [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
 
## 1. Training data
More than 96k human and pathogen protein sequences are given by [bio-datasets](https://pypi.org/project/bio-datasets) package. Before jumping in, the global analysis of the data is always crucial! You can check protein lenght information via ```src/exploratory_data_analysis.py``` with or without histograms.
```bash
python src/exploratory_data_analysis.py
```

You can train/validate/test data and save classifiers as below:
```bash
python src/classifier.py -f probert_embedding # using probert embedding features
python src/classifier.py -f one_hot_encoding # using one-hot encoding features
```
Training with one-hot encoding takes a few minutes the first time but as the feature information will be saved, it will be faster from the 2nd time
You can check the information at any time with the help command:
```bash
python src/classifier.py -h # help
```
The classifiers will be saved in ```checkpoint/```

## 2. Evaluate protein sequences using app
The main class is named ```App``` in ```src/app.py```. 
You can add or modify the protein sequences that you want to evaluate (at the bottom of the code), then just run it:
```bash
python src/app.py
```
The output show the score for each protein and each feature in dictionnary format:
```python
[
  {
    'SGD_probert_embedding':score_of_prot1
    'SGD_one_hot_encoding':score_of_prot1
  },
   {
    'SGD_probert_embedding':score_of_prot2
    'SGD_one_hot_encoding':score_of_prot2
  }
]
```
The score [0,1] correpond to the probability that the proteins belong to the human class.


## Required Python packages
python >= 3.7
```python
numpy
scipy
sklearn
biodatasets
biotransformers
deepchain.components
torch
joblib
loguru
tqdm
statistics
matplotlib
```

