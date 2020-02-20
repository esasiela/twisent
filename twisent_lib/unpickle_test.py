from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


def extract_features_naive(text, B=512):
    """Takes in a text string and maps it to a B-bit hash vector.

    text: str, the text to encode
    B: int, size of the hash vector

    return: np.ndarray vector of size B
    """
    v = np.zeros(B, dtype=np.uint8)
    text = " ".join(text)
    for token in " ".join(text).split():
        v[hash(token) % B] = 1
    return v


def transform_text(X):
    print("transform_text(", type(X), ") shape", X.shape)
    B = 512

    # X_transform = X.apply(extract_features_naive, axis=1)
    # X_transform = np.zeros((X.shape[0], B))
    X_transform = np.array([extract_features_naive(text, B) for text in X.iloc[:, 0]], dtype=np.uint8)

    print("transform_text end:")
    print("\ttype", type(X_transform))
    if hasattr(X_transform, "shape"):
        print("\tshape", X_transform.shape)
    print("First element, type", type(X_transform[0]))
    if hasattr(X_transform[0], "shape"):
        print("\tshape", X_transform[0].shape)

    return X_transform


class DebugMixin:
    def __init__(self, verbose: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    def debug(self, *args, **kwargs):
        name = '.'.join([
            self.__module__,
            self.__class__.__name__
        ])
        if self.verbose:
            print(name, *args, **kwargs)


class FeatureNamesMixin(DebugMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_names = None

    def get_feature_names(self, input_features=None):
        if input_features is not None:
            return input_features
        else:
            return self.feature_names


class CustomHashTransformer(BaseEstimator, TransformerMixin, FeatureNamesMixin):
    def __init__(self, B: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.B = B

    def fit(self, X, y=None):
        """nothing to do at fit() time other than record the feature names"""
        self.debug("X type:", type(X))
        if hasattr(X, "columns"):
            self.debug("saving feature names:", X.columns)
            self.feature_names = X.columns
        else:
            self.debug("X lacks 'columns' attr, cannot save feature names")

        return self

    def transform(self, X):
        self.debug("X type:", type(X))
        X_transform = np.array([extract_features_naive(text, self.B) for text in X.iloc[:, 0]], dtype=np.uint8)
        self.debug("X_transform shape:", X_transform.shape)
        return X_transform


def print_stamp(text: str = "", earlier: datetime = None):
    """
    Prints a timestamp with an optional text message, and an optional duration from an earlier
    datetime

    :return: the current timestamp, so you can pass it back in later to show a duration
    """
    d = datetime.now()
    if earlier is not None:
        print(text, str(d), "elapsed:", str(d - earlier), flush=True)
    else:
        print(text, str(d), flush=True)
    return d


if __name__ == "__main__":
    t = print_stamp("Unpickling model...")
    with open("pickle/twisent_trained_model_lr.pkl", "rb") as f:
        model = pickle.load(f)
    print_stamp("Unpickle complete.", t)
    print("")
    print(model)

    t = print_stamp("Reading full dataframe...")
    full_df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", header=None,
                          names=["target", "status_id", "datetime", "query", "handle", "text"], encoding="latin-1")
    print_stamp("Reading complete", t)

    print("Shape of full_df", full_df.shape)
    print("")

    full_df['target'].replace(4, 1, inplace=True)

    full_df.sort_values("datetime", inplace=True, ascending=True)
    train_size = int(full_df.shape[0] * 0.8)
    test_size = full_df.shape[0] - train_size
    X_train = full_df.drop('target', axis=1).head(train_size)
    y_train = full_df['target'].head(train_size)
    X_test = full_df.drop('target', axis=1).tail(test_size)
    y_test = full_df['target'].tail(test_size)

    # X_train, X_test, y_train, y_test = train_test_split(
    #    full_df.drop('target', axis=1), full_df['target'],
    #    test_size=0.2,
    #    random_state=17, stratify=full_df['target'])

    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    print("")

    t = print_stamp("Predicting on test data...")
    y_pred = model.predict(X_test)
    print_stamp("Prediction complete.", t)
    print("")

    auc = roc_auc_score(y_test, y_pred)
    print("AUC: {0:.05f}".format(auc))
