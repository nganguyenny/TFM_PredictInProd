from re import X
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from datetime import datetime
import pytz
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world 🤗 🚀 🔥"}

@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    X_pred = build_observation(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count)
    print(X_pred)
    print(type(X_pred))
    pipeline = joblib.load('model.joblib')
    y_pred = pipeline.predict(X_pred)
    print(y_pred)
    return {
        'fare': y_pred[0]
    }
    # return {
    #     'pickup_datetime': pickup_datetime,
    #     'pickup_longitude': pickup_longitude,
    #     'pickup_latitude': pickup_latitude,
    #     'dropoff_longitude': dropoff_longitude,
    #     'dropoff_latitude': dropoff_latitude,
    #     'passenger_count': passenger_count
    # }

def build_observation(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    key = "2013-07-06 17:18:00.000000119"
    pickup_datetime = localize_datetime(pickup_datetime)
    pickup_longitude = float(pickup_longitude)
    pickup_latitude = float(pickup_latitude)
    dropoff_longitude = float(dropoff_longitude)
    dropoff_latitude = float(dropoff_latitude)
    passenger_count = int(passenger_count)

    return pd.DataFrame(
        {
        'key': [key],
        'pickup_datetime': [pickup_datetime],
        'pickup_longitude': [pickup_longitude],
        'pickup_latitude': [pickup_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'passenger_count': [passenger_count]
    }
    )


def localize_datetime(pickup_datetime):
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
    return formatted_pickup_datetime

if __name__ == "__main__":
    print(predict(pickup_datetime='2012-10-06 12:10:20',
        pickup_longitude=40.7614327,
        pickup_latitude=-73.9798156,
        dropoff_longitude=40.6413111,
        dropoff_latitude=-73.9797156,
        passenger_count=1))
