import os
import sys
import json

from flask import Flask, request, Response, render_template, redirect, url_for, make_response
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.fields import PasswordField
from wtforms.validators import InputRequired, Length

# needed for the model
from twisent_lib import predictors, spacy_tokenizer, TwisentData, TwitterAccessor
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


@app.route('/')
def welcome():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']
    remote_ip = request.environ.get("HTTP_X_REAL_IP", request.remote_addr)
    print("Remote IP", remote_ip, file=sys.stderr)
    return render_template('index.html', theme=theme, flask_debug=app.debug,
                           twform=TwitterForm(),
                           txform=TextForm(),
                           data=[TwisentData()],
                           active_tab="twitter",
                           ip_blocked=None,
                           username=cookie_username(request))


@app.route('/text', methods=['POST'])
def text():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']

    form = TextForm()
    d = TwisentData()
    if form.validate_on_submit():
        d.text = form.tx.data
        d.pred = meta_model['pipeline'].predict([d.text])[0]
        d.proba = meta_model['pipeline'].predict_proba([d.text])[0, d.pred]
        d.messages.append("TEXT mode")

        return render_template('index.html', theme=theme, flask_debug=app.debug,
                               twform=TwitterForm(),
                               txform=form,
                               data=[d],
                               active_tab="text",
                               ip_blocked=None,
                               username=cookie_username(request))
    else:
        return render_template('index.html', theme=theme, flask_debug=app.debug,
                               twform=TwitterForm(),
                               txform=form,
                               d=[TwisentData()],
                               active_tab="text",
                               ip_blocked=None,
                               username=cookie_username(request))


@app.route('/twitter', methods=['POST'])
def twitter():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']

    form = TwitterForm()
    data_list = []

    if form.validate_on_submit():
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
            data_list.append(d)

        if len(data_list) == 0:
            data_list.append(TwisentData())

        return render_template('index.html', theme=theme, flask_debug=app.debug,
                               twform=form,
                               txform=TextForm(),
                               data=data_list,
                               active_tab="twitter",
                               ip_blocked=None,
                               username=cookie_username(request))
    else:
        return render_template('index.html', theme=theme, flask_debug=app.debug,
                               twform=form,
                               txform=TextForm,
                               data=[TwisentData()],
                               active_tab="twitter",
                               ip_blocked=None,
                               username=cookie_username(request))


@app.route('/pickle', methods=['GET'])
def pickle():
    if not auth_user_verify(request):
        return unauthorized_response(request)

    theme = app.config['THEME']
    d = TwisentData()
    for k, v in meta_model.items():
        d.messages.append("[" + k + "] [" + str(v) + "]")

    return render_template('index.html', theme=theme, flask_debug=app.debug,
                           twform=TwitterForm(),
                           txform=TextForm(),
                           data=[d],
                           active_tab="twitter",
                           ip_blocked=None,
                           username=cookie_username(request))


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
