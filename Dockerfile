FROM ubuntu:14.04

# Update packages
RUN apt-get update -y

# Install Python Setuptools
RUN apt-get install -y python-setuptools

# Install pip
RUN easy_install pip3

# Add and install Python modules
COPY requirements.txt /src/requirements.txt
RUN cd /src; pip3 install -r requirements.txt
RUN cd /src; python3 -m spacy download en_core_web_sm

# Bundle app source
COPY . /src

# Expose
EXPOSE  5000

# Run
CMD ["python", "/src/application.py"]