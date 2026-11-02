"""Phase 1 - Clean the data and create simple features.

What this file does (in order):
  1. Remove extra spaces from all text.
  2. Turn the text "NaN" into a real empty value.
  3. Remove the words stuck in front of numbers: "(min) 24" -> 24, "conditions Sunny" -> "Sunny".
  4. Remove rows with impossible locations (latitude/longitude outside India).
  5. Create new columns: distance_km, hour, weekday.
  6. Keep only the columns we will use, drop rows with missing values.
  7. Save the clean file to data/processed/deliveries_clean.csv

Run from the project folder:
    py src/clean_data.py
"""
import sys
import numpy as np
import pandas as pd
import config as C


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two points on Earth, in km."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def log(step, df):
    print(f"{step:<55} rows left: {len(df)}")


df = pd.read_csv(C.RAW_FILE)
log("Start (raw file)", df)

# 1. Remove extra spaces from every text column ("High " -> "High")
for col in df.columns:
    if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].str.strip()

# 2. Remove the words stuck in front of values
df["Weatherconditions"] = (df["Weatherconditions"]
                           .str.replace("conditions", "", regex=False).str.strip())
df["time_taken_min"] = pd.to_numeric(
    df["Time_taken(min)"].str.replace("(min)", "", regex=False).str.strip(),
    errors="coerce")

# 3. Turn the text "NaN" into a real empty value
for col in df.columns:
    if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].replace("NaN", np.nan)
log("Cleaned text (spaces, words, NaN)", df)

# 4. Remove impossible locations (keep only points inside India's rough box)
lat_ok = lambda s: s.between(6, 38)
lon_ok = lambda s: s.between(68, 98)
good_loc = (lat_ok(df["Restaurant_latitude"]) & lon_ok(df["Restaurant_longitude"]) &
            lat_ok(df["Delivery_location_latitude"]) &
            lon_ok(df["Delivery_location_longitude"]))
df = df[good_loc].copy()
log("Removed invalid locations", df)

# 5. New columns
df["distance_km"] = haversine_km(
    df["Restaurant_latitude"], df["Restaurant_longitude"],
    df["Delivery_location_latitude"], df["Delivery_location_longitude"])

df["hour"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S",
                            errors="coerce").dt.hour
dates = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
if dates.notna().mean() < 0.9:
    print("\nProblem: could not read most Order_Date values.")
    print("Sample values:", list(df["Order_Date"].head(5)))
    print("Please send me this message so I can fix the date format.")
    sys.exit(1)
df["weekday"] = dates.dt.dayofweek          # 0 = Monday ... 6 = Sunday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

# 6. Keep only the columns we will use, and drop rows with missing values
keep = {
    "distance_km": "distance_km",
    "hour": "hour",
    "weekday": "weekday",
    "is_weekend": "is_weekend",
    "Road_traffic_density": "traffic",
    "Weatherconditions": "weather",
    "Type_of_vehicle": "vehicle",
    "multiple_deliveries": "multiple_deliveries",
    "Festival": "festival",
    "time_taken_min": "time_taken_min",
}
df = df[list(keep)].rename(columns=keep)
df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
df = df.dropna()
log("Removed rows with missing values", df)

df["hour"] = df["hour"].astype(int)
df["multiple_deliveries"] = df["multiple_deliveries"].astype(int)
df["time_taken_min"] = df["time_taken_min"].astype(int)

df.to_csv(C.CLEAN_FILE, index=False)
print(f"\nSaved clean file -> {C.CLEAN_FILE}")

# A quick look at the result
print("\nFirst 5 clean rows:")
print(df.head().to_string())
print("\nDistance (km):")
print(df["distance_km"].describe().round(2).to_string())
print("\nDelivery time (min):")
print(df["time_taken_min"].describe().round(1).to_string())
print("\nAverage delivery time by traffic (minutes):")
print(df.groupby("traffic")["time_taken_min"].mean().round(1)
      .sort_values().to_string())
print("\nAverage delivery time by weather (minutes):")
print(df.groupby("weather")["time_taken_min"].mean().round(1)
      .sort_values().to_string())
print("\nPhase 1 done.")
