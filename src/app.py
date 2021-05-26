"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU
          you should choose device='cpu'
"""

import warnings
from typing import Dict, List, Optional

import joblib
import numpy as np
import scipy.sparse as sp
from biotransformers import BioTransformers
from deepchain.components import DeepChainApp
from loguru import logger
from torch import load

warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)

Score = Dict[str, float]
ScoreList = List[Score]


class App(DeepChainApp):
    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.transformer = BioTransformers(backend="protbert", device=device)

        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = None

        # load_model for tensorflow/keras model - load for pytorch model
        if self._checkpoint_filename is not None:
            self.model = load(self.get_checkpoint_path(__file__))

        self.onehot_encoder = joblib.load("checkpoint/onehot_encoder.joblib.pkl")
        self.clf_protbert_embedding = joblib.load(
            "checkpoint/SGD_classifier_feature_probert_embedding.joblib.pkl"
        )
        self.clf_one_hot = joblib.load(
            "checkpoint/SGD_classifier_feature_one_hot_encoding.joblib.pkl"
        )

        # Taking number of unique amino acids thanks to one hot encoder
        self.n_aa = self.onehot_encoder.categories_[0].shape[0]
        # Compute protein max length from classifier and number of amino acids
        self.max_protein_len = self.clf_one_hot.n_features_in_ // self.n_aa

    @staticmethod
    def score_names() -> List[str]:
        """
        Returns:
            List of method names
        """

        return ["SGD_probert_embedding", "SGD_one_hot_encoding"]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """Compute the probability that the proteins belong to the human class using several methods.

        Scores are computed using SGDClassifier (linear classifier with Stochastic Gradient Descent training),
        provided by sklearn.linear_model. Two features are used:
        - Embeddings: given by DeepChain using BioTransformers
        - One-hot encoding: categorical variables (amino acid) are represented as binary vectors using OneHotEncoder.
            First, each protein is represented as a matrix with rows corresponding to its sequence and columns
            corresponding to all 24 unique amino acids found in Bio-dataset. In order to keep the same matrix scale,
            the row number is set to 1024 which corresponds to the embeddings scale. If the protein length is less than
            1024, then we fill in the rest with zero. Otherwise, if the protein is too long, we cut it at the position
            of 1024. Therefore the sum of each row should be 1 or 0 (if there is no amino acid at corresponding
            position).

        Args:
            sequences: list of protein sequences

        Returns:
            list of all proteins score: high score estimates human class, otherwise pathogen class
        """

        logger.info("Feature: embeddings")
        feature_embedding = self.transformer.compute_embeddings(sequences)["cls"]
        score_emb = self.clf_protbert_embedding.predict_proba(feature_embedding)

        logger.info("Feature: one-hot encoding")
        X_one_hot = list()
        for one_seq in sequences:
            one_seq = np.array(list(one_seq))
            if len(one_seq) > self.max_protein_len:
                one_seq = one_seq[: self.max_protein_len]
            encoded = self.onehot_encoder.transform(one_seq.reshape(-1, 1))

            # if len(protein) < max_protein_len, fill the rest by 0
            encoded_pad = sp.csr_matrix((self.max_protein_len, self.n_aa))
            encoded_pad[: encoded.shape[0], : encoded.shape[1]] = encoded
            encoded = encoded_pad
            encoded = encoded.reshape(1, self.max_protein_len * self.n_aa).tocsr()
            X_one_hot.append(encoded)
        X = sp.vstack(X_one_hot)
        score_onehot = self.clf_one_hot.predict_proba(X)

        score_list = [
            {self.score_names()[0]: s_emb[0], self.score_names()[1]: s_onehot[0]}
            for s_emb, s_onehot in zip(score_emb, score_onehot)
        ]

        return score_list


if __name__ == "__main__":

    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]
    app = App("cpu")
    scores = app.compute_scores(sequences)
    logger.info(scores)
