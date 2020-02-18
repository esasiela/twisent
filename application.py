import os
import sys
import json

from flask import Flask, request, Response, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


# Create the Flask app, load default config from config.py, load secret config from instance/config.py
app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config')
app.config.from_pyfile('config.py')


class TwisentForm(FlaskForm):
    t = StringField("Input", validators=[DataRequired()])
    submit = SubmitField("Predict Sentiment")


class TwisentData:
    msg = ""


@app.route('/')
def welcome():
    theme = app.config['THEME']
    return render_template('index.html', theme=theme, flask_debug=app.debug, form=TwisentForm(), data=None)


@app.route('/twisent', methods=['POST'])
def twisent():
    print("twisent()", flush=True)
    theme = app.config['THEME']

    form = TwisentForm()
    if form.validate_on_submit():
        # they gave a non-empty text field
        print("twisent(), validation passed", flush=True)
        print("twisent(), t=", form.t, "[", form.t.data, "]", flush=True)

        d = TwisentData()
        d.msg = "i have text"
        return render_template('index.html', theme=theme, flask_debug=app.debug, form=form, data=d)
        # return redirect(url_for('index'))
    else:
        return render_template('index.html', theme=theme, flask_debug=app.debug, form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
