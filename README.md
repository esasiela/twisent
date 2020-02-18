# TwiSent

This repo is the home of a Twitter sentiment analyzer using machine learning and Python.

The system consists of the following modules:

1. Twitter scraper
1. Preprocessor
1. Model Trainer
1. Trained Model
1. Sentiment prediction
1. Output
1. User Interface

## High Level View

The 80,000 foot view of Twisent:

1. Either using GUI or CLI, user supplies input
1. Massage the input to determine if it is an @username, status ID, URL, or #hashtag
1. Scraper retrieves tweet data using Twitter API
1. For each tweet retrieved, preprocess the text
1. Text is passed through the trained model to predict the sentiment
1. Output is written

In order to provide this flow, analysis and training is done during development time:

1. Obtain training data
1. Visualize training data
1. Trial and error of a few basic models and preprocessors
1. Gridsearch to refine hyperparameters of best model identified above
1. Store the trained final best model 

### Input

Input processing follows this sequence:

* **@username** - input starts with `@`, retrieve recent messages from that user
* **#hashtag** - input starts with `#`, retrieve recent messages referencing the specified hashtag
* **Twitter URL** - input starts with `http`, parse the `status` query parameter for a single tweet
* **Status ID** -  input is all numeric digits, retrieve the matching single tweet

### Output

Using the Twitter API, the standard developer account may only search for 7 days of history, so if Tweets are missing, this may be the cause.

* .csv file with list of post data scraped from the page or Twitter API.
* keywords of the post
* predicted emotion of the post

