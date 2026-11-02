"""Turns a clean delivery table into numbers the model can read.

The model can only read numbers, so words like "Jam" or "Sunny" must be converted.
This file is shared: Phase 2 uses it to train, and Phase 3 will use it again
for the Ajmer deliveries, so both always speak the same 'number language'.
"""
import numpy as np
import pandas as pd

# Traffic has a natural order: Low < Medium < High < Jam
TRAFFIC_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

# Weather and vehicle have no order, so each value becomes its own 0/1 column.
WEATHER_VALUES = ["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"]
VEHICLE_VALUES = ["motorcycle", "scooter", "electric_scooter", "bicycle"]


def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["distance_km"] = df["distance_km"]

    # Hour is a circle (23:00 is next to 00:00), so we use sin and cos.
    X["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    X["weekday"] = df["weekday"]
    X["is_weekend"] = df["is_weekend"]
    X["multiple_deliveries"] = df["multiple_deliveries"]
    X["festival_yes"] = (df["festival"] == "Yes").astype(int)
    X["traffic_level"] = df["traffic"].map(TRAFFIC_ORDER)

    for w in WEATHER_VALUES:
        X["weather_" + w] = (df["weather"] == w).astype(int)
    for v in VEHICLE_VALUES:
        X["vehicle_" + v] = (df["vehicle"] == v).astype(int)
    return X
