import os
from datetime import datetime
from spacy.lang.en import English
import spacy
import string
from sklearn.base import TransformerMixin
from twitter import Api
from urllib3.request import urlencode


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


class TwisentData:
    """Used to store data about a tweet and/or text sentiment prediction"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = []
        self.text = None
        self.tweet = None
        self.pred = -1
        self.proba = -1


class TwitterAccessor:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tweets = []
        # TODO error handling if Api call fails
        self.api = Api(
            consumer_key=os.environ.get("TWITTER_CONSUMER_KEY"),
            consumer_secret=os.environ.get("TWITTER_CONSUMER_SECRET"),
            access_token_key=os.environ.get("TWITTER_ACCESS_TOKEN_KEY"),
            access_token_secret=os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
        )

    def get_tweet_by_id(self, status_id):
        # TODO error handling if Api call fails
        self.tweets.append(self.api.GetStatus(status_id=status_id))

    def get_tweets_by_username(self, username):
        # TODO error handling if Api call fails
        self.tweets = self.api.GetUserTimeline(screen_name=username, count=5)

    def get_tweets_by_hashtag(self, hashtag: str):
        hashtag = hashtag.split()[0]
        query_string = urlencode({'q': str("(" + hashtag + ")")})
        # Twitter API returns stuff without the hashtag, so filter only relevants
        # TODO error handling if Api call fails
        search_result = self.api.GetSearch(raw_query=query_string, count=5)
        for tweet in search_result:
            # hashtags is a list, each element has a 'text' attribute
            for h in tweet.hashtags:
                if hashtag.lower() == str("#" + h.text.lower()):
                    self.tweets.append(tweet)
                    break


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