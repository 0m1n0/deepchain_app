import argparse
import warnings
from os import path

import joblib
import numpy as np
import scipy.sparse as sp
from biodatasets import load_dataset
from loguru import logger
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)


class Classifier:
    def __init__(self):
        self.pathogen = load_dataset("pathogen")
        self.X, self.y = self.pathogen.to_npy_arrays(
            input_names=["sequence"], target_names=["class"]
        )
        self.X = self.X[0]
        self.y = self.y[0]
        self.clf = None
        pass

    def one_hot_encoder(self, one_seq, onehot_encoder, n_aa, max_protein_len):
        # returns max(protein_len) x len(all unique aa) array
        one_seq = np.array(list(one_seq))
        # if len(protein) > 1024, then cut
        if len(one_seq) > max_protein_len:
            one_seq = one_seq[:max_protein_len]
        encoded = onehot_encoder.transform(one_seq.reshape(-1, 1))

        # if len(protein) < max(protein_len), fill the rest by 0
        encoded_pad = sp.csr_matrix((max_protein_len, n_aa))
        encoded_pad[: encoded.shape[0], : encoded.shape[1]] = encoded
        return encoded_pad

    def get_features_onehot(self, max_protein_len):
        # one-hot encoding to represent protein sequences
        file_name = "data/onehot_sp_matrix.npz"
        if path.exists(file_name):
            logger.warning("Loading precomputed one-hot encoding features.")
            X = sp.load_npz(file_name)
        else:
            aa_list = list(set("".join(self.X)))
            aa_list.sort()

            onehot_encoder = OneHotEncoder(categories="auto", sparse=True)
            onehot_encoder.fit(np.array(aa_list).reshape(-1, 1))
            joblib.dump(
                onehot_encoder,
                f"checkpoint/onehot_encoder.joblib.pkl",
                compress=9,
            )

            X_one_hot = list()
            for one_seq in tqdm(self.X):
                one_hot = self.one_hot_encoder(
                    one_seq, onehot_encoder, len(aa_list), max_protein_len
                )
                # one_hot = one_hot.reshape(1, max_protein_len * len(aa_list)).tocsr()
                one_hot = one_hot.reshape(1, -1).tocsr()
                X_one_hot.append(one_hot)
            X = sp.vstack(X_one_hot)
            sp.save_npz(file_name, X)
        return X

    def get_features_embedding_probert(self):
        embed = self.pathogen.get_embeddings("sequence", "protbert", "cls")
        return embed

    def split_data(self, features):
        # train = 60%
        # validation = 20%
        # test = 20%
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, self.y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )  # 0.25 x 0.8 = 0.2
        return X_train, X_train_val, X_val, y_train, y_train_val, y_val

    def run_sgd(self, X_train, X_train_val, X_val, y_train, y_train_val, y_val):
        # If you want to fit a large-scale linear classifier without copying a
        # dense numpy C-contiguous double precision array as input, we suggest
        # to use the SGDClassifier class instead. The objective function can
        # be configured to be almost the same as the LinearSVC model.

        # create a SGDClassifier
        self.clf = SGDClassifier(loss="log", random_state=42)
        # train the model
        self.clf.fit(X_train, y_train)

        # validation
        y_val_pred = self.clf.predict(X_val)
        logger.info(f"Validation accuracy: {metrics.accuracy_score(y_val, y_val_pred)}")
        # test
        y_test_pred = self.clf.predict(X_train_val)
        logger.info(
            f"Test accuracy: {metrics.accuracy_score(y_train_val, y_test_pred)}"
        )

    def save(self, path):
        if self.clf is None:
            raise Exception("You should run run_sgd before saving.")

        joblib.dump(
            self.clf,
            path,
            compress=9,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn and save classifier.")
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        choices=["probert_embedding", "one_hot_encoding"],
        default="one_hot_encoding",
        help="Select feature for classifier.",
    )
    parser.add_argument(
        "-fs",
        "--features_size",
        type=int,
        default=1024,
        help="Set size of features for one hot encoding (default=1024).",
    )
    args = parser.parse_args()

    classifier = Classifier()
    if args.feature == "one_hot_encoding":
        features = classifier.get_features_onehot(args.features_size)
    else:
        features = classifier.get_features_embedding_probert()

    X_train, X_train_val, X_val, y_train, y_train_val, y_val = classifier.split_data(
        features
    )
    classifier.run_sgd(X_train, X_train_val, X_val, y_train, y_train_val, y_val)
    classifier.save(f"checkpoint/SGD_classifier_feature_{args.feature}.joblib.pkl")
