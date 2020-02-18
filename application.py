import os
import sys
import json

from flask import Flask, request, Response, render_template


# Default config vals
THEME = 'default' if os.environ.get('THEME') is None else os.environ.get('THEME')
FLASK_DEBUG = 'false' if os.environ.get('FLASK_DEBUG') is None else os.environ.get('FLASK_DEBUG')

# Create the Flask app
application = Flask(__name__)

# Load config values specified above
application.config.from_object(__name__)

# Load configuration vals from a file
application.config.from_envvar('APP_CONFIG', silent=True)

# Only enable Flask debugging if an env var is set to true
application.debug = application.config['FLASK_DEBUG'] in ['true', 'True']


@application.route('/')
def welcome():
    theme = application.config['THEME']
    return render_template('index.html', theme=theme, flask_debug=application.debug)


@application.route('/signup', methods=['POST'])
def signup():
    signup_data = dict()
    for item in request.form:
        signup_data[item] = request.form[item]

    try:
        store_in_dynamo(signup_data)
        publish_to_sns(signup_data)
    except ConditionalCheckFailedException:
        return Response("", status=409, mimetype='application/json')

    return Response(json.dumps(signup_data), status=201, mimetype='application/json')


def publish_to_sns(signup_data):
    try:
        sns_conn.publish(application.config['NEW_SIGNUP_TOPIC'], json.dumps(signup_data),
                         "New signup: %s" % signup_data['email'])
    except Exception as ex:
        sys.stderr.write("Error publishing subscription message to SNS: %s" % ex.message)


if __name__ == '__main__':
    application.run(host='0.0.0.0')
