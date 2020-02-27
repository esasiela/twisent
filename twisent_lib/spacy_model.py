from datetime import datetime
import string
import pickle

import spacy

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from twisent_lib import print_stamp, twisent_tokenizer, CleanTextTransformer


if __name__ == "__main__":
    t = print_stamp("Reading full dataframe...")
    full_df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", header=None, names=["target", "status_id", "datetime", "query", "handle", "text"], encoding="latin-1")
    print_stamp("Reading complete.", t)

    retrainWhole = True
    truncateRows = 0

    if truncateRows > 0:
        # grab the first bunch of rows, then grab another bunch starting at 800000
        # this gets us the earliest bunch of pos and earliest bunch of negs
        full_df = full_df.head(truncateRows).append(full_df.iloc[800000:800000+truncateRows, :])

    # convert target=4 to target=1
    full_df['target'].replace(4, 1, inplace=True)
    # sort chrono
    full_df.sort_values("datetime", ascending=True, inplace=True)

    print("")
    print("full_df shape", full_df.shape)
    print("")
    print("full_df info", full_df.info())
    print("")
    print(full_df.head())
    print("")
    print("target label distribution")
    print(full_df['target'].value_counts(dropna=False))

    train_size = int(full_df.shape[0] * 0.8)
    test_size = full_df.shape[0] - train_size
    X_train = full_df['text'].head(train_size)
    y_train = full_df['target'].head(train_size)
    X_test = full_df['text'].tail(test_size)
    y_test = full_df['target'].tail(test_size)
    print("")
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print(y_train.value_counts())
    print("")
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    print(y_test.value_counts())
    print("")

    #nlp = spacy.load('en')
    nlp = spacy.load('en_core_web_sm')

    bow_vector = CountVectorizer(tokenizer=twisent_tokenizer, ngram_range=(1, 1), encoding="latin-1")
    #tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

    model_lr = LogisticRegression(
        max_iter=5000,
        solver="sag"
    )

    model_xgb = xgb.XGBClassifier(
        objective="binary:logistic",
        eta=0.3,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=5,
        reg_alpha=1.5,
        min_child_weight=1.0,
    )

    model_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        max_features="sqrt"
    )

    model = model_lr

    pipe = Pipeline([
        ("cleaner", CleanTextTransformer()),
        ("vectorizer", bow_vector),
        ("classifier", model),
    ])

    t = print_stamp("Training model...")
    pipe.fit(X_train, y_train)
    print_stamp("Training complete.", t)
    print("")
    t = print_stamp("Predicting...")
    #y_pred = pipe.predict(X_test)
    y_pred = pipe.predict(X_test)
    print_stamp("Predicting complete.", t)

    print("")
    print("y_pred shape", y_pred.shape)
    print("y_test shape", y_test.shape)

    print("")
    print("Metrics:")
    print("\tAccuracy : {0:.05f}".format(accuracy_score(y_test, y_pred)))
    print("\tPrecision: {0:.05f}".format(precision_score(y_test, y_pred)))
    print("\tRecall   : {0:.05f}".format(recall_score(y_test, y_pred)))
    print("\tROC AUC  : {0:.05f}".format(roc_auc_score(y_test, y_pred)))

    # retrain on whole dataset
    if retrainWhole:
        print("")
        t = print_stamp("Retraining on full dataset...")
        pipe.fit(full_df['text'], full_df['target'])
        print_stamp("Retraining complete.", t)
        print("")

        t = print_stamp("Saving model to pickle file...")
        pickle_path = "pickle/twisent_trained_model" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pkl"
        print(pickle_path)

        meta_pickle = {
            'pipeline': pipe,
            'num_train_rows': full_df.shape[0],
            'train_date': datetime.now(),
            'git_commit': '47f3d4e77d8b7eb0d78fa6297d12bbbbc12a19a6',
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(meta_pickle, f)
        print_stamp("Pickle complete.", t)
        print("")

        t = print_stamp("Verifying pickle...")
        with open(pickle_path, "rb") as f:
            loaded_meta = pickle.load(f)
        print(loaded_meta)
        print_stamp("Verification complete.", t)


