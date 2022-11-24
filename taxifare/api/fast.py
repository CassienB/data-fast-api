from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from taxifare.ml_logic.preprocessor import preprocess_features
import pandas as pd
from taxifare.ml_logic.registry import load_model
from taxifare.ml_logic.params import COLUMN_NAMES_RAW
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(pickup_datetime: datetime,  # 2013-07-06 17:18:00
            pickup_longitude: float,    # -73.950655
            pickup_latitude: float,     # 40.783282
            dropoff_longitude: float,   # -73.984365
            dropoff_latitude: float,    # 40.769802
            passenger_count: int):      # 1
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """
    X_pred = pd.DataFrame(dict(
            key=[str(pickup_datetime)],  # useless but the pipeline requires it
            pickup_datetime=[pickup_datetime],
            pickup_longitude=[float(pickup_longitude)],
            pickup_latitude=[float(pickup_latitude)],
            dropoff_longitude=[float(dropoff_longitude)],
            dropoff_latitude=[float(dropoff_latitude)],
            passenger_count=[int(passenger_count)]
        ))
    X_processed = preprocess_features(X_pred)

    model = load_model()
    y_pred = model.predict(X_processed)

    return {'fare_amount':float(y_pred)}


@app.get("/")
def root():
    return {'greeting': 'Hello'}
