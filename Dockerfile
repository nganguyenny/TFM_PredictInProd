FROM python:3.8.6-buster

# the trained model
COPY api /api
# the code of the project which is required in order to load the model
COPY TaxiFareModel /TaxiFareModel
# the code of our API
COPY model.joblib /model.joblib
# the list of requirements
COPY requirements.txt /requirements.txt

#COPY /Users/nganguyen/code/gcp/data-lewagon-85edaa0a5c93.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
