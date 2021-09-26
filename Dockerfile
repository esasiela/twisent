FROM python:3.8

WORKDIR "/src"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Update spacy language pack
RUN python -m spacy download en_core_web_sm

# WSGI server
RUN pip install gunicorn

# Copy application files into image
COPY . .
COPY ./pickle/twisent_trained_model.pkl pickle/

# Expose
EXPOSE  5000

# CMD ["python", "/src/application.py"]
CMD ["gunicorn", "--bind=0.0.0.0:5000", "application:app"]
