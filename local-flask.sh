#!/bin/sh

. ./venv/Scripts/activate

. ./instance/twisent_config_secret.sh

flask run
