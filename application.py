import os
import sys
import json

from flask import Flask, request, Response, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# needed for the model
from twisent_lib import predictors, spacy_tokenizer
import pickle


# Create the Flask app, load default config from config.py, load secret config from instance/config.py
app = Flask(__name__)
app.config.from_object('config')
for k, v in app.config.items():
    if os.environ.get(k) is not None:
        app.config[k] = os.environ.get(k)

# global variable, suitable for demo purposes, not production
meta_model = None
#with open("pickle/twisent_trained_model.pkl", "rb") as f:
#    meta_model = pickle.load(f)


class TwisentForm(FlaskForm):
    t = StringField("Input", validators=[DataRequired()])
    submit = SubmitField("Predict Sentiment")


class TwisentData:
    msg = ""


@app.route('/')
def welcome():
    theme = app.config['THEME']
    d = TwisentData()
    d.msg = "working directory: " + os.getcwd() + "\n" + str(os.listdir(os.getcwd()))
    return render_template('index.html', theme=theme, flask_debug=app.debug, form=TwisentForm(), data=d)


@app.route('/twisent', methods=['POST'])
def twisent():
    #print("twisent()", flush=True)
    theme = app.config['THEME']

    form = TwisentForm()
    if form.validate_on_submit():
        # they gave a non-empty text field
        #print("twisent(), validation passed", flush=True)
        #print("twisent(), t=", form.t, "[", form.t.data, "]", flush=True)

        pred = meta_model['pipeline'].predict([form.t.data])
        proba = meta_model['pipeline'].predict_proba([form.t.data])
        d = TwisentData()
        d.msg = "prediction: {0:d}, {1:.05f}".format(pred[0], proba[0, pred[0]])

        return render_template('index.html', theme=theme, flask_debug=app.debug, form=form, data=d)
        # return redirect(url_for('index'))
    else:
        return render_template('index.html', theme=theme, flask_debug=app.debug, form=form)


@app.route('/pickle', methods=['GET'])
def pickle():
    theme = app.config['THEME']
    d = TwisentData()
    d.msg = str(meta_model)

    return render_template('index.html', theme=theme, flask_debug=app.debug, form=TwisentForm(), data=d)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
