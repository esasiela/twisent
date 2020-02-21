FROM python:3
#FROM ubuntu:14.04

# Update packages
#RUN apt-get update -y

# Install Python Setuptools
#RUN apt-get install -y python3-setuptools python3-pip

# Install pip
# RUN easy_install pip

# Add and install Python modules
COPY requirements.txt /src/requirements.txt
RUN cd /src; pip install -r requirements.txt
RUN cd /src; python -m spacy download en_core_web_sm

# Bundle app source
COPY . /src
COPY ./pickle/twisent_trained_model.pkl /src/pickle/

# Expose
EXPOSE  5000

# Run
CMD ["python", "/src/application.py"]
