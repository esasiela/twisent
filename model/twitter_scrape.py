from twitter import Api


def get_tweets_by_user(api=None, screen_name=None):
    timeline = api.GetUserTimeline(screen_name=screen_name, count=1)
    earliest_tweet = min(timeline, key=lambda x: x.id).id
    print("Earliest tweet in timeline:", earliest_tweet)

    return timeline


def get_tweet_by_id(api: Api = None, status_id=None):
    return api.GetStatus(status_id=status_id)



if __name__ == "__main__":
    api = Api(consumer_key="W7oddRnrNR90nwZzKmd1FNCFz",
                      consumer_secret="6u4G7GdAc5bQmUg06KYJkYDqLh9rO8JsFsBzmcARsg9aeVrxlk",
                      access_token_key="1229455506621763584-XFZfG8Cw6KubEf2frran49fnkZaeRw",
                      access_token_secret="xQN3yQG1bRdiKxPmjcnQyDk8XdMGZoRPpjuTxy7zD3jzj")

    print("Verifying Twitter OAuth credentials:", api.VerifyCredentials())

    tweet = get_tweet_by_id(api, 1227715403251638273)
    print(type(tweet))
    print(tweet)
    print(tweet.text)

    #timeline = get_tweets_by_user(api=api, screen_name="arduino")
    #print("Found {0:d} tweets.".format(len(timeline)))
    #for tweet in timeline:
    #    print(type(tweet))
    #    print(tweet)
    #    print(tweet.text)
