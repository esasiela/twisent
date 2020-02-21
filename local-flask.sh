#!/bin/sh

export DISPLAY_APP_NAME="Hedge Court - TwiSent (DEV)"
export PICKLE_PATH="pickle/twisent_trained_model.pkl"
export FLASK_APP=application.py
export FLASK_ENV=development
flask run
