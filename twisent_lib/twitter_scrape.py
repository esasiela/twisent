from twitter import Api
import os
from twisent_lib import TwitterAccessor

def get_tweets_by_user(api=None, screen_name=None):
    timeline = api.GetUserTimeline(screen_name=screen_name, count=1)
    print("Timeline type", type(timeline))

    earliest_tweet = min(timeline, key=lambda x: x.id).id
    print("Earliest tweet in timeline:", earliest_tweet)

    return timeline


def get_tweet_by_id(api: Api = None, status_id=None):
    return api.GetStatus(status_id=status_id)


if __name__ == "__main__":
    ta = TwitterAccessor()
    ta.get_tweets_by_hashtag("#arduino")

    for tweet in ta.tweets:
        print("=========================")
        for k, v in tweet.AsDict().items():
            print("k=[", k, "] v=[", v, "]")

    #api = Api(consumer_key=os.environ.get("TWITTER_CONSUMER_KEY"),
    #          consumer_secret=os.environ.get("TWITTER_CONSUMER_SECRET"),
    #          access_token_key=os.environ.get("TWITTER_ACCESS_TOKEN_KEY"),
    #          access_token_secret=os.environ.get("TWITTER_ACCESS_TOKEN_SECRET"))
    #
    #print("Verifying Twitter OAuth credentials:", api.VerifyCredentials())
    #
    ##tweet = get_tweet_by_id(api, 1227715403251638273)
    #tweet = get_tweets_by_user(api, "@arduino")
    #print(type(tweet))
    #print(tweet)
    #print(tweet.text)
    #
    #print("")
    #
    #for k, v in tweet.AsDict().items():
    #    print("k=[", k, "] v=[", v, "]")
    #    if k == "created_at":
    #        print("CREATED_AT!!!!")

    #timeline = get_tweets_by_user(api=api, screen_name="arduino")
    #print("Found {0:d} tweets.".format(len(timeline)))
    #for tweet in timeline:
    #    print(type(tweet))
    #    print(tweet)
    #    print(tweet.text)
