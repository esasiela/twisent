import os
import sys
import json

from flask import Flask, request, Response, render_template, redirect, url_for, make_response
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, IntegerField
from wtforms.fields import PasswordField
from wtforms.validators import InputRequired, Length, NumberRange

# needed for the model
from twisent_lib import TwisentData, TwitterAccessor
import pickle


# Create the Flask app, load default config from config.py, load secret config from instance/config.py
app = Flask(__name__)
app.config.from_object('config')
for k, v in app.config.items():
    if os.environ.get(k) is not None:
        app.config[k] = os.environ.get(k)

# global variable, suitable for demo purposes, not production
meta_model = None
# "pickle/twisent_trained_model.pkl"
with open(app.config['PICKLE_PATH'], "rb") as f:
    meta_model = pickle.load(f)


class TwitterForm(FlaskForm):
    tw = StringField("Input", validators=[InputRequired()])
    throttle = IntegerField("Result Limit", validators=[NumberRange(min=1, max=50)])
    submit = SubmitField("Predict Sentiment")


class TextForm(FlaskForm):
    tx = TextAreaField("Input", validators=[InputRequired(), Length(max=280)])
    submit = SubmitField("Predict Sentiment")


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[InputRequired()])
    password = PasswordField("Password", validators=[InputRequired()])
    submit = SubmitField("Login")


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


def auth_user_verify(request):
    return cookie_username(request) is not None


def unauthorized_response(request):
    return render_template('index.html', theme=app.config['THEME'], flask_debug=app.debug,
                           twform=TwitterForm(),
                           txform=TextForm(),
                           data=[TwisentData()],
                           active_tab="text",
                           ip_blocked=None,
                           username=None)


def cookie_username(request):
    c = request.cookies.get("magic")
    if c is not None:
        u, p = c.split(":")
        if p == os.environ.get("MAGIC_WORD"):
            return u
    else:
        return None


class TwisentDisplay:
    """
    Holds the display data needed to render the page
    """
    def __init__(self, twform: TwitterForm, txform: TextForm, username: str,
                 active_tab: str = "twitter", ip_blocked: str = None, pickle: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.twform = twform
        self.txform = txform
        self.username = username
        self.data = []
        self.active_tab = active_tab
        self.ip_blocked = ip_blocked
        self.pickle = pickle
        self.messages = []
        print("TwisentDisplay constructor, len(data)", len(self.data), "len(messages)", len(self.messages), file=sys.stderr)

    def get_count_by_label(self, label=-1):
        """
        Returns the number of items with the indicated label.  Default -1 means total, 0=Neg, 1=Pos
        :param label:
        :return: int count of items
        """
        if label == -1:
            return len(self.data)
        else:
            return sum(1 for d in self.data if d.pred == label)

    def get_proba_by_label(self, label=-1):
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
        elif label == -1:
            # weird case, change neg's to 1-proba, which is different than rest of display
            pos_proba = sum(d.proba for d in self.data if d.pred == 1)
            neg_proba = sum(1 - d.proba for d in self.data if d.pred == 0)
            return (pos_proba + neg_proba) / len(self.data)
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


@app.route('/')
def welcome():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']
    # remote_ip = request.environ.get("HTTP_X_REAL_IP", request.remote_addr)
    # print("Remote IP", remote_ip, file=sys.stderr)
    tw = TwitterForm()
    tw.throttle.data = TwitterAccessor.COUNT_THROTTLE
    display = TwisentDisplay(tw, TextForm(), username=cookie_username(request))
    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/text', methods=['POST'])
def text():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']

    form = TextForm()
    tw = TwitterForm()
    tw.throttle.data = TwitterAccessor.COUNT_THROTTLE
    display = TwisentDisplay(tw, form, username=cookie_username(request), active_tab="text")
    d = TwisentData()
    if form.validate_on_submit():
        d.text = form.tx.data
        d.pred = meta_model['pipeline'].predict([d.text])[0]
        d.proba = meta_model['pipeline'].predict_proba([d.text])[0, d.pred]

        display.messages.append("TEXT mode")
        display.data.append(d)

    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/twitter', methods=['POST'])
def twitter():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']

    form = TwitterForm()
    display = TwisentDisplay(form, TextForm(), username=cookie_username(request))

    if form.validate_on_submit():
        TwitterAccessor.COUNT_THROTTLE = form.throttle.data

        t = form.tw.data
        ta = TwitterAccessor()

        # first, look at t to see if it is a URL that we can do something with:
        # hashtag https://twitter.com/hashtag/Arduino?src=hashtag_click
        # user    https://twitter.com/arduino
        # tweet   https://twitter.com/arduino/status/1225785143501230082
        url_hashtag_prefix = "https://twitter.com/hashtag/"
        url_status_delim = "/status/"
        url_username_prefix = "https://twitter.com/"
        if t.startswith(url_hashtag_prefix):
            # kill the prefix, split on ? and keep the left half
            print("URL Search, hashtag", t, file=sys.stderr)
            t = "#" + t.replace(url_hashtag_prefix, "").split("?")[0]
            form.tw.data = t
        elif url_status_delim in t:
            # split by /status/, grab the right half, split that by ? and grab the left half
            # more fun this way than a regex
            # TODO check that status id is purely numeric, or just let it fail because this URL paste is kinda get-what-you-get
            print("URL Search, status", t, file=sys.stderr)
            t = t.split(url_status_delim)[1].split("?")[0]
            form.tw.data = t
        elif t.startswith(url_username_prefix) and "/" not in t.replace(url_username_prefix, ""):
            # all we have is a single path element on the URL, that is the username
            # kill the prefix, split by ? and keep the left half
            print("URL Search, user", t, file=sys.stderr)
            t = "@" + t.replace(url_username_prefix, "").split("?")[0]
            form.tw.data = t

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
            # assume text is pure numeric, i.e. a status id
            status_id = int(t)
            ta.get_tweet_by_id(status_id)
            # TODO error handling (Api fails, no tweet found, etc)

        for tweet in ta.tweets:
            d = TwisentData()
            d.text = tweet.AsDict()['text']
            d.tweet = tweet
            d.pred = meta_model['pipeline'].predict([d.text])[0]
            d.proba = meta_model['pipeline'].predict_proba([d.text])[0, d.pred]
            d.messages.append("MODE: twitter - " + input_mode)
            display.data.append(d)

    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/pickle', methods=['GET'])
def pickle():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']

    tw = TwitterForm()
    tw.throttle.data = TwitterAccessor.COUNT_THROTTLE
    display = TwisentDisplay(tw, TextForm(), username=cookie_username(request), pickle=True)

    for k, v in meta_model.items():
        display.messages.append("[" + k + "] [" + str(v) + "]")

    return render_template('index.html', theme=theme, flask_debug=app.debug, display=display)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        if form.password.data == os.environ.get("MAGIC_WORD"):
            res = make_response(
                render_template('login.html', theme=app.config['THEME'], flask_debug=app.debug,
                                form=form, msg="Successful Login", username=form.username.data)
            )
            res.set_cookie("magic", str(form.username.data) + ":" + str(form.password.data),
                           max_age=(60 * 60 * 24 * 365 * 2))
            return res
        else:
            # they submittited a bad password
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
