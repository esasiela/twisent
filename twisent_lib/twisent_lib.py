import os
from datetime import datetime
import string
from urllib3.request import urlencode

import pandas as pd

from spacy.lang.en import English
import spacy

from sklearn.base import TransformerMixin

from twitter import Api
from twitter.models import Status, User


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
        self.spacy_text = []

    def get_spacy_text(self):
        if len(self.spacy_text) == 0:
            self.spacy_text = twisent_tokenizer(self.text)
        return self.spacy_text

    def as_dataframe(self):
        """
        returns a one-row dataframe with this objects data in it.
        :return: dataframe
        """
        if self.tweet is not None:
            my_tweet = self.tweet
            hashtag_text = ""
            for h in self.tweet.hashtags:
                hashtag_text += " #" + h.text
        else:
            my_tweet = Status()
            my_tweet.user = User()
            my_tweet.created_at = datetime.now()
            hashtag_text = ""

        d = {
            "pos proba": [self.proba if self.pred == 1 else 1 - self.proba],
            "time": [my_tweet.created_at],
            "status_id": [my_tweet.id],
            "screen_name": [my_tweet.user.screen_name],
            "user_name": [my_tweet.user.name],
            "hashtags": [hashtag_text],
            "favorite_count": [my_tweet.favorite_count],
            "retweet_count": [my_tweet.retweet_count],
            "text": [self.text],
            "keywords": [":".join(self.get_spacy_text())]
        }
        return pd.DataFrame(d)

    def get_csv_string(self):
        """
        returns a csv-escaped string with the following fields.  If text mode (i.e. not a tweet) then all
        twitter-specific fields will be emptystring.

        pos-proba
        time
        status id
        user handle
        user screen name
        hashtags (delimit with #)
        favorite count
        retweet count
        raw text (csv escaped)
        keywords (delimit with #)

        :return: csv-escaped string
        """
        return self.as_dataframe().to_csv(index=False)


class TwitterAccessor:
    COUNT_THROTTLE = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tweets = []
        self.api = Api(
            consumer_key=os.environ.get("TWITTER_CONSUMER_KEY"),
            consumer_secret=os.environ.get("TWITTER_CONSUMER_SECRET"),
            access_token_key=os.environ.get("TWITTER_ACCESS_TOKEN_KEY"),
            access_token_secret=os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
        )

    def get_tweet_by_id(self, status_id):
        self.tweets.append(self.api.GetStatus(status_id=status_id))

    def get_tweets_by_username(self, username):
        self.tweets = self.api.GetUserTimeline(screen_name=username, count=TwitterAccessor.COUNT_THROTTLE)

    def get_tweets_by_hashtag(self, hashtag: str):
        hashtag = hashtag.split()[0]
        query_string = urlencode({'q': str("(" + hashtag + ")")})
        # Twitter API returns stuff without the hashtag, so filter only relevants
        search_result = self.api.GetSearch(raw_query=query_string, count=TwitterAccessor.COUNT_THROTTLE)
        for tweet in search_result:
            # hashtags is a list, each element has a 'text' attribute
            for h in tweet.hashtags:
                if hashtag.lower() == str("#" + h.text.lower()):
                    # only keep tweets that actually have this hashtag, Twitter API sometimes sends others
                    self.tweets.append(tweet)
                    break

    def get_tweets_by_search(self, query: str):
        query_string = urlencode({'q': str("(" + query + ")")})
        self.tweets = self.api.GetSearch(raw_query=query_string, count=TwitterAccessor.COUNT_THROTTLE)

    def get_tweets_by_geo(self, lat, lng, radius):
        geo_str = "{0:s},{1:s},{2:s}".format(lat, lng, radius)
        self.tweets = self.api.GetSearch(geocode=geo_str, count=TwitterAccessor.COUNT_THROTTLE)


def twisent_tokenizer(sentence, parser=English(), stop_words=spacy.lang.en.stop_words.STOP_WORDS,
                      punctuation=string.punctuation):
    """
    Breaks the sentence into tokens and applies the following preprocessing:
    1) strip whitespace
    2) lower case
    3) lemmatize
    4) run preprocess_token to really dig in and clean it up

    :param sentence: the input string to be tokenized
    :param parser: the language parser (default=spacy.English())
    :param stop_words: list of str
    :param punctuation: str containing all punctuation marks
    :return: list of str tokens
    """
    tokens = parser(sentence)

    # strip, lower, lemmatize each token
    tokens = [word.lemma_.lower().strip() for word in tokens]

    # clean the lemma up
    tokens = [preprocess_token(word, stop_words, punctuation) for word in tokens]

    # remove blanks
    return [word for word in tokens if word != ""]


def preprocess_token(t: str, stop_words=spacy.lang.en.stop_words.STOP_WORDS, punctuation=string.punctuation):
    """
    Processes the token.  Assumes it has been lowered, stripped, and lemmatized.
    Removes the following:
    1) begins with @ (username mentions)
    2) punctuation
    3) stop words
    4) begins with http (urls)
    5) 'rt' (all retweets begin with this token, just adds noise)
    :param t: the token to parse
    :param stop_words: stop words to remove
    :param punctuation: punctuation to remove
    :return: str, the final token
    """
    # remove 'begins with @'
    if t.startswith('@'):
        return ""

    # remove punctuation
    t = t.translate(str.maketrans('', '', punctuation))

    # remove stop words, ^http, ^rt$
    if t in stop_words or t.startswith("http") or t == "rt":
        return ""

    return t


def clean_text(text):
    return text.strip().lower()


class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
