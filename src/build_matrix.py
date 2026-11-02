"""Phase 3 - Build the 'travel-time map' for the Ajmer delivery points.

The route optimizer (next phases) needs to know, instantly, how many minutes it
takes to go from any stop to any other stop. This file makes that table.

Steps:
  1. Read your Ajmer locations file and check it carefully.
  2. Measure the distance between every pair of stops.
  3. Ask the trained model how long a trip takes under the conditions below.
  4. Test whether the model really takes longer for longer distances.
  5. Save the distance table, the time table and a picture of the points.

Run from the project folder:
    py src/build_matrix.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from features import make_features

# ------------------------------------------------------------------
# Settings you can change: the situation we plan routes for
# ------------------------------------------------------------------
HOUR = 18                 # 18 = 6 PM (evening rush)
WEEKDAY = 4               # 0 = Monday ... 4 = Friday ... 6 = Sunday
TRAFFIC = "High"          # Low, Medium, High or Jam
WEATHER = "Sunny"         # Sunny, Cloudy, Fog, Stormy, Sandstorms, Windy
VEHICLE = "motorcycle"    # motorcycle, scooter, electric_scooter, bicycle
MULTIPLE_DELIVERIES = 1   # other deliveries the rider is carrying (0-3)
FESTIVAL = "No"           # "Yes" or "No"

BASE_SPEED_KMPH = 25      # ASSUMED normal city speed of a bike in calm traffic
FORCE_METHOD = None       # None = decide automatically, or "model" / "hybrid"
# ------------------------------------------------------------------

DIST_FILE = C.ROOT / "data" / "processed" / "distance_matrix.csv"
TIME_FILE = C.ROOT / "data" / "processed" / "time_matrix.csv"
STOPS_FILE = C.ROOT / "data" / "processed" / "ajmer_stops_clean.csv"
FEATURES_FILE = C.ROOT / "models" / "eta_features.json"


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two points on Earth, in km."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def stop(msg):
    print("\nProblem:", msg)
    raise SystemExit(1)


# ---------- 1. Read and check the Ajmer file ----------
if not C.AJMER_FILE.exists():
    stop(f"{C.AJMER_FILE} not found. Put ajmer_locations.csv in data/raw/")
if not C.MODEL_FILE.exists() or not FEATURES_FILE.exists():
    stop("Model not found. Run Phase 2 first:  py src/train_eta.py")

loc = pd.read_csv(C.AJMER_FILE)
if "coordinates" in loc.columns:
    coords = loc["coordinates"].astype(str).str.replace("°", "", regex=False)
    empty = loc["coordinates"].isna() | (coords.str.strip() == "")
    if empty.any():
        names = list(loc.loc[empty, "name"])
        stop(f"{empty.sum()} rows have no coordinates yet, for example: {names[:5]}\n"
             "Fill the 'coordinates' column (copy from Google Maps) and save as CSV again.")
    parts = coords.str.split(",", n=1, expand=True)
    if parts.shape[1] < 2:
        stop("The coordinates must look like  26.4499, 74.6399  (two numbers with a comma).")
    loc["lat"] = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    loc["lon"] = pd.to_numeric(parts[1].str.strip(), errors="coerce")
elif {"latitude", "longitude"} <= set(loc.columns):
    loc["lat"], loc["lon"] = loc["latitude"], loc["longitude"]
else:
    stop("Could not find a 'coordinates' column in ajmer_locations.csv")

bad = loc["lat"].isna() | loc["lon"].isna()
if bad.any():
    stop("Could not read the coordinates of: " + str(list(loc.loc[bad, "name"])[:5]) +
         "\nEach one must look like  26.4499, 74.6399  (two numbers separated by a comma).")
if len(loc) < 4:
    stop("Please fill at least 1 depot and 3 customers (20 customers is ideal).")

swapped = loc["lat"] > loc["lon"]
if swapped.any():
    stop("Latitude and longitude look swapped for: " + str(list(loc.loc[swapped, "name"])[:5]) +
         "\nFor Ajmer, latitude is about 26 and longitude is about 74.")
for col, lo, hi in [("lat", 6, 38), ("lon", 68, 98)]:
    if not loc[col].between(lo, hi).all():
        stop(f"Some {col} values are outside India. Please re-check the coordinates.")

loc["demand"] = pd.to_numeric(loc.get("demand", 1), errors="coerce").fillna(1).astype(int)
loc.loc[0, "demand"] = 0   # first row is the depot
print(f"Loaded {len(loc)-1} customers + 1 depot ('{loc.loc[0, 'name']}').")

# ---------- 2. Distance between every pair ----------
lat, lon = loc["lat"].to_numpy(), loc["lon"].to_numpy()
dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
names = list(loc["name"])
print(f"Farthest pair of stops: {dist.max():.1f} km. "
      f"Farthest stop from depot: {dist[0].max():.1f} km.")
if dist[0].max() > 30:
    print("Warning: a stop is more than 30 km from the depot. Is it inside Ajmer?")

# ---------- 3. Ask the model ----------
model = xgb.XGBRegressor()
model.load_model(str(C.MODEL_FILE))
feature_names = json.loads(FEATURES_FILE.read_text())


def predict_minutes(distances_km, traffic=TRAFFIC, weather=WEATHER, hour=HOUR):
    d = pd.DataFrame({"distance_km": np.asarray(distances_km, dtype=float)})
    d["hour"], d["weekday"] = hour, WEEKDAY
    d["is_weekend"] = int(WEEKDAY >= 5)
    d["traffic"], d["weather"], d["vehicle"] = traffic, weather, VEHICLE
    d["multiple_deliveries"], d["festival"] = MULTIPLE_DELIVERIES, FESTIVAL
    return model.predict(make_features(d)[feature_names])


# ---------- 4. Does the model take longer for longer distances? ----------
test_km = [1, 2, 3, 5, 8, 10, 12, 15]
test_min = predict_minutes(test_km)
print("\nTest: same conditions, only the distance changes")
for k, m in zip(test_km, test_min):
    print(f"   {k:>2} km  ->  model says {m:5.1f} min")

extra = float(predict_minutes([12])[0] - predict_minutes([2])[0])
speed = 10 / (extra / 60) if extra > 0 else float("inf")
weak = speed > 45          # 10 extra km in under ~13 min is not realistic in a city
print(f"\n10 extra km add {extra:.1f} min  (= {speed:.0f} km/h, "
      f"{'too fast for a city: distance effect is WEAK' if weak else 'realistic'})")

method = FORCE_METHOD or ("hybrid" if weak else "model")

# ---------- 5. Build the time table ----------
if method == "model":
    print("\nMethod used: MODEL. Every trip time comes straight from the model.")
    t = predict_minutes(dist.ravel()).reshape(dist.shape)
else:
    # The model learns how much traffic, weather and hour slow things down.
    # We measure that slowdown against a calm situation and apply it to distance / speed.
    ref = predict_minutes([8], traffic="Low", weather="Sunny", hour=11)[0]
    now = predict_minutes([8])[0]
    factor = now / ref
    print("\nMethod used: HYBRID.")
    print(f"   The model says current conditions are {factor:.2f}x slower than a calm "
          f"Low-traffic, sunny, 11 AM trip.")
    print(f"   Trip time = distance / {BASE_SPEED_KMPH} km/h x {factor:.2f}")
    t = dist / BASE_SPEED_KMPH * 60 * factor
t = np.asarray(t, dtype=float)
np.fill_diagonal(t, 0)

# ---------- 6. Save ----------
C.RESULTS_DIR.mkdir(exist_ok=True)
C.CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(dist, index=names, columns=names).round(3).to_csv(DIST_FILE)
pd.DataFrame(t, index=names, columns=names).round(2).to_csv(TIME_FILE)
loc[["name", "lat", "lon", "demand"]].to_csv(STOPS_FILE, index=False)

plt.figure(figsize=(6, 6))
plt.scatter(loc["lon"][1:], loc["lat"][1:], s=40)
for i in range(1, len(loc)):
    plt.annotate(str(i), (loc["lon"][i], loc["lat"][i]), fontsize=8,
                 xytext=(3, 3), textcoords="offset points")
plt.scatter(loc["lon"][0], loc["lat"][0], c="red", marker="*", s=250, label="Depot")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Delivery points in Ajmer"); plt.legend(); plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "ajmer_points.png", dpi=150)
plt.close()

print("\nTravel time in minutes (first 5 stops):")
print(pd.DataFrame(t[:5, :5], index=names[:5], columns=[n[:10] for n in names[:5]])
      .round(1).to_string())
print(f"\nSaved: {TIME_FILE.name}, {DIST_FILE.name}, {STOPS_FILE.name} (data/processed)")
print("Saved picture: results/ajmer_points.png")
print("\nPhase 3 done.")
