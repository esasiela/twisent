# TwiSent

This repo is the home of a Twitter sentiment analyzer using machine learning and Python.

Try it out in [production](https://twisent.hedgecourt.com/), running on AWS EC2 Elastic Beanstalk.

View the docs in my [Hedge Court Software Engineering Portfolio](https://www.hedgecourt.com/portfolio/twisent/) page.

The portfolio page is better, but here's a brief overview...

## Module List

The system consists of the following modules:

1. Twitter API access
1. Model Trainer
1. Sentiment prediction
1. User Interface

## High Level View

The 80,000 foot view of Twisent:

1. Model code is in `twisent_lib` directory
1. Training and Pipeline is in `twisent_lib/spacy_model.py`
1. Flask UI code is in `application.py`
1. Flask UI gets the trained model from `pickle/twisent_trained_model.pkl`

## Prediction Flow

When the UI receives user input:
1. Massage the input to determine if it is an @username, status ID, URL, or #hashtag
1. Scraper retrieves tweet data using Twitter API
1. For each tweet retrieved, preprocess the text
1. Text is passed through the trained model to predict the sentiment
1. Output is written in pretty and CSV form

In order to provide this flow, analysis and training is done during development time:

1. Obtain training data from Sentiment140

### Input Processing

Input processing follows this sequence:

* **@username** - input starts with `@`, retrieve recent messages from that user
* **#hashtag** - input starts with `#`, retrieve recent messages referencing the specified hashtag
* **Twitter URL** - input starts with `http`, parse the `status` query parameter for a single tweet
* **Status ID** -  input is all numeric digits, retrieve the matching single tweet
