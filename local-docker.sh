#!/bin/sh

. ./instance/twisent_config_secret.sh

docker run \
  --env TWITTER_CONSUMER_KEY=$TWITTER_CONSUMER_KEY \
  --env TWITTER_CONSUMER_SECRET=$TWITTER_CONSUMER_SECRET \
  --env TWITTER_ACCESS_TOKEN_KEY=$TWITTER_ACCESS_TOKEN_KEY \
  --env TWITTER_ACCESS_TOKEN_SECRET=$TWITTER_ACCESS_TOKEN_SECRET \
  --env FLASK_ENV=production \
  --env APPLICATION_ROOT=/app/twisent \
  --env DEBUG=False \
  -p 5001:5000 \
  --name hc-twisent \
  hc-twisent

#  --env SCRIPT_NAME=/hello \
