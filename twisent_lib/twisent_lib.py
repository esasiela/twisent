from datetime import datetime
from spacy.lang.en import English
import spacy
import string
from sklearn.base import TransformerMixin


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


def spacy_tokenizer(sentence, parser=English(), stop_words=spacy.lang.en.stop_words.STOP_WORDS,
                    punctuations=string.punctuation):
    mytokens = parser(sentence)

    # lemmatize each token and convert tolower
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # remove stopwords
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    return mytokens


def clean_text(text):
    return text.strip().lower()


class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}