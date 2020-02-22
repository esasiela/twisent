import os
import sys
import json

from flask import Flask, request, Response, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
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


@app.route('/')
def welcome():
    theme = app.config['THEME']
    # d = TwisentData()
    # d.msg = "working directory: " + os.getcwd() + "\n" + str(os.listdir(os.getcwd()))
    return render_template('index.html', theme=theme, flask_debug=app.debug,
                           twform=TwitterForm(),
                           txform=TextForm(),
                           data=[TwisentData()],
                           active_tab="twitter")


@app.route('/text', methods=['POST'])
def text():
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
                               active_tab="text")
    else:
        return render_template('index.html', theme=theme, flask_debug=app.debug,
                               twform=TwitterForm(),
                               txform=form,
                               d=[TwisentData()],
                               active_tab="text")


@app.route('/twitter', methods=['POST'])
def twitter():
    theme = app.config['THEME']

    form = TwitterForm()
    data_list = []

    if form.validate_on_submit():
        t = form.tw.data
        ta = TwitterAccessor()

        if t.startswith('@'):
            input_mode = "@handle"
            ta.get_tweets_by_handle(form.tw.data)
        else:
            input_mode = "status_id"
            # assume text is pure numeric, i.e. a status id
            status_id = int(form.tw.data)
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
                               active_tab="twitter")
    else:
        return render_template('index.html', theme=theme, flask_debug=app.debug,
                               twform=form,
                               txform=TextForm,
                               data=[TwisentData()],
                               active_tab="twitter")


@app.route('/pickle', methods=['GET'])
def pickle():
    theme = app.config['THEME']
    d = TwisentData()
    for k, v in meta_model.items():
        d.messages.append("[" + k + "] [" + str(v) + "]")

    return render_template('index.html', theme=theme, flask_debug=app.debug,
                           twform=TwitterForm(),
                           txform=TextForm(),
                           data=[d],
                           active_tab="twitter")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
