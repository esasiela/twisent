import os
import sys
from random import random
import pprint

import urllib.parse

from flask import Flask, request, Response, render_template, redirect, url_for, make_response
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, IntegerField
from wtforms.fields import PasswordField, HiddenField
from wtforms.validators import InputRequired, Length, NumberRange

# needed for the model
from twisent_lib import TwisentData, TwitterAccessor, TwisentLog
import pickle

from twitter import TwitterError


# Create the Flask app, load default config from config.py, load secret config from instance/config.py
app = Flask(__name__, static_url_path="/app/twisent/static")
app.config.from_object('config')
for k, v in app.config.items():
    if os.environ.get(k) is not None:
        app.config[k] = os.environ.get(k)


if app.config["DUMP_CONFIG"]:
    print("Dumping Config...")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(app.config)
else:
    print("Suppressing config dump (set DUMP_CONFIG=True to enable")


# global variable, suitable for demo purposes, not production
meta_model = None
# "pickle/twisent_trained_model.pkl"
with open(app.config['PICKLE_PATH'], "rb") as f:
    meta_model = pickle.load(f)

logger = TwisentLog(app.config["TWISENT_LOG_URL"], app.config["TWISENT_LOG_ENABLE"])
logger.log("boot", "--sys--")


def ip_whitelist_verify(request):
    remote_addr = request.environ.get("HTTP_X_REAL_IP", request.remote_addr)
    white_list = os.environ.get("IP_WHITELIST")
    retval = request.environ.get("HTTP_X_REAL_IP", request.remote_addr) in \
             os.environ.get("IP_WHITELIST").replace(" ", "").split(",")
    if not retval:
        print("Blocking IP:", remote_addr, "WHITELIST:", white_list, "returning:", retval, file=sys.stderr)
    return retval


def ip_whitelist_response(request):
    return render_template('index.html', theme=app.config['THEME'], flask_debug=app.debug,
                           twform=TwitterForm(),
                           txform=TextForm(),
                           data=[TwisentData()],
                           active_tab="twitter",
                           ip_blocked=request.environ.get("HTTP_X_REAL_IP", request.remote_addr),
                           username=None)


def cookie_username(request):
    c = request.cookies.get("magic")
    if c is not None:
        u, p = c.split(":")
        if p == os.environ.get("MAGIC_WORD"):
            return u
    # return None
    return "Guest"


class TwitterForm(FlaskForm):
    tw = StringField("Input", validators=[InputRequired()])
    throttle = IntegerField("Result Limit", validators=[NumberRange(min=1, max=50)])
    submit = SubmitField("Predict Sentiment")


class GeoForm(FlaskForm):
    lat = HiddenField("Latitude", validators=[InputRequired()], id="geo-lat", default=app.config['GEO_DEFAULT_LAT'])
    lng = HiddenField("Longitude", validators=[InputRequired()], id="geo-lng", default=app.config['GEO_DEFAULT_LNG'])
    radius = HiddenField("Radius", validators=[InputRequired()], id="geo-rad", default=app.config['GEO_DEFAULT_RADIUS'])
    submit = SubmitField("Predict Sentiment")


class TextForm(FlaskForm):
    tx = TextAreaField("Input", validators=[InputRequired(), Length(max=280)])
    submit = SubmitField("Predict Sentiment")


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[InputRequired()])
    password = PasswordField("Password", validators=[InputRequired()])
    submit = SubmitField("Login")


class TwisentDisplay:
    """
    Holds the display data needed to render the page
    """
    def __init__(self, twform: TwitterForm, geoform: GeoForm, txform: TextForm, username: str,
                 active_tab: str = "twitter", ip_blocked: str = None, pickle: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.twform = twform
        self.geoform = geoform
        self.txform = txform
        self.username = username
        self.data = []
        self.active_tab = active_tab
        self.ip_blocked = ip_blocked
        self.pickle = pickle
        self.messages = []
        self.errors = []
        # print("TwisentDisplay constructor, len(data)", len(self.data), "len(messages)", len(self.messages), file=sys.stderr)

    def cachebuster(self):
        if app.config["CACHEBUSTER"] == "RANDOM":
            return str(random())[2:]
        else:
            return app.config["CACHEBUSTER"]

    def get_count_by_label(self, label=None):
        """
        Returns the number of items with the indicated label.  Default -1 means total, 0=Neg, 1=Pos
        :param label:
        :return: int count of items
        """
        if label is None:
            return len(self.data)
        else:
            return sum(1 for d in self.data if d.pred == label)

    def get_proba_by_label(self, label=None):
        """
        Returns the average proba for the given label.
        :param label:
        :return: float probability 0..1
        """
        if self.get_count_by_label(label) == 0:
            if label == 0:
                # REMEMBER: this is a display only, not a math model, in display we sub neg from 1, so return 1 to get zero
                return 1
            else:
                return 0
        elif len(self.data) - self.get_count_by_label(-1) == 0:
            # they're all unpredictable
            return 0
        elif label is None:
            # weird case, change neg's to 1-proba, which is different than rest of display
            pos_proba = sum(d.proba for d in self.data if d.pred == 1)
            neg_proba = sum(1 - d.proba for d in self.data if d.pred == 0)
            return (pos_proba + neg_proba) / (len(self.data) - self.get_count_by_label(-1))
        else:
            return sum(d.proba for d in self.data if d.pred == label) / self.get_count_by_label(label)

    def get_csv_string(self):
        """
        Returns a csv representation of all search results
        :return:
        """
        df = None
        for d in self.data:
            if df is None:
                df = d.as_dataframe()
            else:
                df = df.append(d.as_dataframe())

        if df is None:
            return ""
        else:
            return df.to_csv(index=False)


def auth_user_verify(request):
    return cookie_username(request) is not None


def unauthorized_response(request):
    tw = TwitterForm()
    tw.throttle.data = TwitterAccessor.COUNT_THROTTLE
    display = TwisentDisplay(tw, GeoForm(), TextForm(), username=cookie_username(request),
                             active_tab="text")

    return render_template('index.html', theme=app.config['THEME'], flask_debug=app.debug,
                           display=display)


@app.route('/app/twisent')
def welcome():
    logger.log("home", cookie_username(request))

    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']
    # remote_ip = request.environ.get("HTTP_X_REAL_IP", request.remote_addr)
    # print("Remote IP", remote_ip, file=sys.stderr)
    tw = TwitterForm()
    tw.throttle.data = TwitterAccessor.COUNT_THROTTLE
    display = TwisentDisplay(tw, GeoForm(), TextForm(), username=cookie_username(request))
    display.messages.append("Pickle Path [{0:s}]".format(app.config['PICKLE_PATH']))
    display.messages.append("CWD [{0:s}]".format(os.getcwd()))

    for k, v in app.config.items():
        display.messages.append("key [{0:s}] val [{1:s}]".format(k, str(v)))

    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/app/twisent/text', methods=['POST'])
def text():
    #if not auth_user_verify(request):
    #    return unauthorized_response(request)

    theme = app.config['THEME']

    form = TextForm()
    tw = TwitterForm()
    tw.throttle.data = TwitterAccessor.COUNT_THROTTLE
    display = TwisentDisplay(tw, GeoForm(), form, username=cookie_username(request), active_tab="text")
    d = TwisentData()
    if form.validate_on_submit():
        d.text = form.tx.data
        if len(d.get_spacy_text()) == 0:
            d.pred = -1
            d.proba = -1
        else:
            d.pred = meta_model['pipeline'].predict([d.text])[0]
            d.proba = meta_model['pipeline'].predict_proba([d.text])[0, d.pred]

        display.messages.append("TEXT mode")
        display.data.append(d)

    logger.log("text", cookie_username(request), {"input": form.tx.data})

    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/app/twisent/twitter', methods=['POST'])
def twitter():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']

    form = TwitterForm()
    display = TwisentDisplay(form, GeoForm(), TextForm(), username=cookie_username(request))

    log_data = {}

    if form.validate_on_submit():
        TwitterAccessor.COUNT_THROTTLE = form.throttle.data

        t = form.tw.data
        log_data["input"] = t
        ta = TwitterAccessor()

        # first, look at t to see if it is a URL that we can do something with:
        # search  https://twitter.com/search?q=%40arduino&src=typed_query
        # hashtag https://twitter.com/hashtag/Arduino?src=hashtag_click
        # user    https://twitter.com/arduino
        # tweet   https://twitter.com/arduino/status/1225785143501230082
        url_search_prefix = "https://twitter.com/search?q="
        url_hashtag_prefix = "https://twitter.com/hashtag/"
        url_status_delim = "/status/"
        url_username_prefix = "https://twitter.com/"

        if t.startswith(url_search_prefix):
            # kill the prefix, split on & and keep the first element, and pass it to the rest
            print("URL Search, search", t, file=sys.stderr)
            log_data["url_search"] = "True"
            t = urllib.parse.unquote(t.replace(url_search_prefix, "").split("&")[0])
            form.tw.data = t
        elif t.startswith(url_hashtag_prefix):
            # kill the prefix, split on ? and keep the left half
            print("URL Search, hashtag", t, file=sys.stderr)
            log_data["hashtag_search"] = "True"
            t = "#" + t.replace(url_hashtag_prefix, "").split("?")[0]
            form.tw.data = t
        elif url_status_delim in t:
            # split by /status/, grab the right half, split that by ? and grab the left half
            # more fun this way than a regex
            print("URL Search, status", t, file=sys.stderr)
            log_data["status_search"] = "True"
            t = t.split(url_status_delim)[1].split("?")[0]
            form.tw.data = t
        elif t.startswith(url_username_prefix) and "/" not in t.replace(url_username_prefix, ""):
            # all we have is a single path element on the URL, that is the username
            # kill the prefix, split by ? and keep the left half
            print("URL Search, user", t, file=sys.stderr)
            log_data["user_search"] = "True"
            t = "@" + t.replace(url_username_prefix, "").split("?")[0]
            form.tw.data = t

        try:
            if t.startswith('@'):
                # print("Twitter search, mode=@username", t, file=sys.stderr)
                input_mode = "@username"
                ta.get_tweets_by_username(t)
            elif t.startswith('#'):
                # print("Twitter search, mode=#hashtag", t, file=sys.stderr)
                input_mode = "#hashtag"
                ta.get_tweets_by_hashtag(t)
            else:
                # print("Twitter search, mode=status_id", t, file=sys.stderr)
                input_mode = "status_id"
                # if text is pure numeric (i.e. a status id) then perform a status id search
                try:
                    status_id = int(t)
                    log_data["status_id"] = str(status_id)
                    ta.get_tweet_by_id(status_id)
                except ValueError:
                    log_data["keyword_search"] = "true"
                    ta.get_tweets_by_search(t)

        except TwitterError as e:
            log_data["twitter_error"] = e.message
            display.errors.append(e.message)

        for tweet in ta.tweets:
            d = TwisentData()
            d.text = tweet.AsDict()['text']
            d.tweet = tweet
            if len(d.get_spacy_text()) == 0:
                d.pred = -1
                d.proba = -1
            else:
                d.pred = meta_model['pipeline'].predict([d.text])[0]
                d.proba = meta_model['pipeline'].predict_proba([d.text])[0, d.pred]
            d.messages.append("MODE: twitter - " + input_mode)
            display.data.append(d)

    log_data["input_mode"] = input_mode
    logger.log("twitter", cookie_username(request), log_data)

    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/app/twisent/geo', methods=['POST'])
def geo():
    #print("GEO route", file=sys.stderr)
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']

    form = GeoForm()
    tw = TwitterForm()
    log_data = {}
    tw.throttle.data = TwitterAccessor.COUNT_THROTTLE
    display = TwisentDisplay(tw, form, TextForm(), username=cookie_username(request), active_tab="geo")
    if form.validate_on_submit():
        log_data["lat"] = form.lat.data
        log_data["lng"] = form.lng.data
        display.messages.append("GEO mode")
        display.messages.append("GEO: lat={0:s} lng={1:s} rad={2:s}".format(form.lat.data, form.lng.data, form.radius.data))
        ta = TwitterAccessor()
        try:
            ta.get_tweets_by_geo(form.lat.data, form.lng.data, form.radius.data)
        except TwitterError as e:
            log_data["twitter_error"] = e.message
            display.errors.append(e.message)

        for tweet in ta.tweets:
            d = TwisentData()
            d.text = tweet.AsDict()['text']
            d.tweet = tweet
            if len(d.get_spacy_text()) == 0:
                d.pred = -1
                d.proba = -1
            else:
                d.pred = meta_model['pipeline'].predict([d.text])[0]
                d.proba = meta_model['pipeline'].predict_proba([d.text])[0, d.pred]
            display.data.append(d)
    else:
        display.errors.append("Click map to specify location.")
        #display.messages.append("Geo sub, failed validation")

    logger.log("geo", cookie_username(request), log_data)

    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/app/twisent/pickle', methods=['GET'])
def pickle():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']

    tw = TwitterForm()
    tw.throttle.data = TwitterAccessor.COUNT_THROTTLE
    display = TwisentDisplay(tw, GeoForm(), TextForm(), username=cookie_username(request), pickle=True)

    for k, v in meta_model.items():
        display.messages.append("[" + k + "] [" + str(v) + "]")

    logger.log("pickle", cookie_username(request))

    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/app/twisent/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        if form.password.data == os.environ.get("MAGIC_WORD"):
            logger.log("login_success", form.username.data)
            res = make_response(
                render_template('login.html', theme=app.config['THEME'], flask_debug=app.debug,
                                form=form, msg="Successful Login", username=form.username.data)
            )
            res.set_cookie("magic", str(form.username.data) + ":" + str(form.password.data),
                           max_age=(60 * 60 * 24 * 365 * 2))
            return res
        else:
            # they submitted a bad password
            logger.log("login_fail", form.username.data)
            res = make_response(
                render_template('login.html', theme=app.config['THEME'], flask_debug=app.debug,
                                form=form, msg="Invalid Password", username=None)
            )
            res.set_cookie("magic", "failure:failure", max_age=0)
            return res
    else:
        return render_template('login.html', theme=app.config['THEME'], flask_debug=app.debug,
                               form=form, msg=None, username=cookie_username(request))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
