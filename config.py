# override in environment variables (used to be 'instance/config.py')
FLASK_DEBUG = False
FLASK_ENV = "development"

SECRET_KEY = "override me in env var"

DUMP_CONFIG = True

TWISENT_LOG_ENV = "override me in env var"
TWISENT_LOG_URL = "override me in env var"
TWISENT_LOG_ENABLE = False

# set to "RANDOM" in dev to always reload CSS, override in ENV with a static version number
CACHEBUSTER = "RANDOM"

THEME = "default"
DISPLAY_APP_NAME = "CHANGE_ME Hedge Court - TwiSent"
DISPLAY_PAGE_MSG = 0

SHOW_PICKLE = False
PICKLE_PATH = "/src/pickle/twisent_trained_model.pkl"

IP_WHITELIST = "override me in env var"
MAGIC_WORD = "override me in env var"

TWITTER_CONSUMER_KEY = "override me in env var"
TWITTER_CONSUMER_SECRET = "override me in env var"
TWITTER_ACCESS_TOKEN_KEY = "override me in env var"
TWITTER_ACCESS_TOKEN_SECRET = "override me in env var"

# Bell Hall, where it all started...
GEO_DEFAULT_LAT = 43.0016724
GEO_DEFAULT_LNG = -78.7894114
GEO_DEFAULT_RADIUS = "1km"

