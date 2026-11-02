"""Phase 7 (part 1) - Explainability: WHY does the model predict slow trips?

We use SHAP values (TreeSHAP, built into XGBoost, so no extra library needed).
A SHAP value says: 'compared with an average trip in the training data, this
feature pushed the predicted time UP or DOWN by this many minutes'.

We group related columns so the story is readable:
  hour_sin + hour_cos  -> 'hour of day'
  weather_*            -> 'weather'
  vehicle_*            -> 'vehicle type'   ... and so on.

Outputs (results/):
  shap_global.png        what matters most for the model overall
  shap_scenario.png      what is slowing / speeding the CURRENT plan (hour, traffic, ...)
  shap_example_trip.png  breakdown of one long trip
  route_explanations.txt plain-English explanation for each bike's route

Run from the project folder (after Phase 6):
    py src/explain.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import routing_core as R
from features import make_features

# ------------------------------------------------------------------
# KEEP THESE SAME AS IN build_matrix.py (the situation we plan for)
# ------------------------------------------------------------------
HOUR = 18
WEEKDAY = 4
TRAFFIC = "High"
WEATHER = "Sunny"
VEHICLE = "motorcycle"
MULTIPLE_DELIVERIES = 1
FESTIVAL = "No"
# ------------------------------------------------------------------

FEATURES_FILE = C.ROOT / "models" / "eta_features.json"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def stop(msg):
    print("\nProblem:", msg)
    raise SystemExit(1)


for f in (C.MODEL_FILE, FEATURES_FILE, C.CLEAN_FILE):
    if not f.exists():
        stop(f"{f} not found. Run Phase 1 and Phase 2 first.")

model = xgb.XGBRegressor()
model.load_model(str(C.MODEL_FILE))
feature_names = json.loads(FEATURES_FILE.read_text())
booster = model.get_booster()
p = R.load_problem()

# routes to explain: GA+SA if available, else GA
routes_file = C.RESULTS_DIR / "sa_routes.json"
if not routes_file.exists():
    routes_file = C.RESULTS_DIR / "ga_routes.json"
if not routes_file.exists():
    stop("No routes found. Run Phase 5 (and 6) first.")
routes = json.load(open(routes_file))["routes"]
print(f"Explaining routes from {routes_file.name}")

# ---------- helpers ----------
GROUPS = {
    "distance": ["distance_km"],
    "hour of day": ["hour_sin", "hour_cos"],
    "traffic": ["traffic_level"],
    "weather": [c for c in feature_names if c.startswith("weather_")],
    "vehicle type": [c for c in feature_names if c.startswith("vehicle_")],
    "day of week": ["weekday", "is_weekend"],
    "extra deliveries carried": ["multiple_deliveries"],
    "festival": ["festival_yes"],
}
GROUPS = {g: [c for c in cols if c in feature_names] for g, cols in GROUPS.items()}
GROUPS = {g: cols for g, cols in GROUPS.items() if cols}
GROUP_NAMES = list(GROUPS)


def make_df(dists):
    d = pd.DataFrame({"distance_km": np.asarray(dists, dtype=float)})
    d["hour"], d["weekday"] = HOUR, WEEKDAY
    d["is_weekend"] = int(WEEKDAY >= 5)
    d["traffic"], d["weather"], d["vehicle"] = TRAFFIC, WEATHER, VEHICLE
    d["multiple_deliveries"], d["festival"] = MULTIPLE_DELIVERIES, FESTIVAL
    return d


def shap_groups(X):
    """Return (minutes pushed by each group [n x groups], base value, predictions)."""
    c = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    vals, base = c[:, :-1], c[:, -1]
    idx = {name: i for i, name in enumerate(X.columns)}
    G = np.column_stack([vals[:, [idx[col] for col in cols]].sum(axis=1)
                         for cols in GROUPS.values()])
    return G, float(base.mean()), vals.sum(axis=1) + base


def barh(ax_names, values, title, path, signed=False):
    order = np.argsort(np.abs(values))
    names = [ax_names[i] for i in order]
    vals = np.asarray(values)[order]
    colors = ["tab:red" if v > 0 else "tab:green" for v in vals] if signed else "tab:blue"
    plt.figure(figsize=(7, 4))
    plt.barh(names, vals, color=colors)
    if signed:
        plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("minutes (red = slower, green = faster)" if signed else "average effect (minutes)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------- 1. Global: what matters most overall ----------
clean = pd.read_csv(C.CLEAN_FILE)
bg = clean.sample(min(1000, len(clean)), random_state=C.SEED)
Xbg = make_features(bg)[feature_names]
G_bg, base, _ = shap_groups(Xbg)
global_imp = np.abs(G_bg).mean(axis=0)
barh(GROUP_NAMES, global_imp, "What the travel-time model cares about (overall)",
     C.RESULTS_DIR / "shap_global.png")
print(f"\nAverage training trip takes about {base:.1f} min (the SHAP starting point).")
print("Most important factors overall:")
for i in np.argsort(-global_imp)[:5]:
    print(f"   {GROUP_NAMES[i]:26s} average effect {global_imp[i]:.2f} min")

# ---------- 2. Legs of the planned routes ----------
legs, leg_route = [], []
for k, r in enumerate(routes):
    path = [0] + list(r) + [0]
    for a, b in zip(path[:-1], path[1:]):
        legs.append(float(p.D[a, b]))
        leg_route.append(k)
legs, leg_route = np.array(legs), np.array(leg_route)
Xl = make_features(make_df(legs))[feature_names]
G_l, _, pred_l = shap_groups(Xl)

# ---------- 3. Scenario: what slows the plan right now ----------
scenario = G_l.mean(axis=0)
barh(GROUP_NAMES, scenario, f"Why trips are slow/fast now ({HOUR}:00, {TRAFFIC} traffic, {WEATHER})",
     C.RESULTS_DIR / "shap_scenario.png", signed=True)

# what-if: calm reference
calm = make_df([8.0]); calm["hour"], calm["traffic"], calm["weather"] = 11, "Low", "Sunny"
now_t = float(model.predict(make_features(make_df([8.0]))[feature_names])[0])
calm_t = float(model.predict(make_features(calm)[feature_names])[0])
print(f"\nAn 8 km trip: {now_t:.1f} min now vs {calm_t:.1f} min in calm conditions "
      f"(Low traffic, sunny, 11 AM).")

# ---------- 4. One example trip ----------
i = int(np.argmax(legs))
barh(GROUP_NAMES, G_l[i], f"Longest trip ({legs[i]:.1f} km): predicted {pred_l[i]:.1f} min",
     C.RESULTS_DIR / "shap_example_trip.png", signed=True)

# ---------- 5. Plain-English text per route ----------
VALUE_TEXT = {
    "hour of day": f"{HOUR}:00", "traffic": f"{TRAFFIC} traffic", "weather": WEATHER,
    "vehicle type": VEHICLE, "day of week": DAYS[WEEKDAY],
    "extra deliveries carried": f"{MULTIPLE_DELIVERIES} extra deliveries", "festival": f"festival: {FESTIVAL}",
}
lines = [f"Routes explained: {routes_file.name}",
         f"Conditions: {DAYS[WEEKDAY]} {HOUR}:00, {TRAFFIC} traffic, {WEATHER}, {VEHICLE}",
         f"Reference: an average training trip takes {base:.1f} min.", ""]
for k, r in enumerate(routes):
    if not r:
        continue
    m = leg_route == k
    g = G_l[m].mean(axis=0)
    top = np.argsort(-np.abs(g))[:3]
    parts = []
    for j in top:
        name = GROUP_NAMES[j]
        what = f"{legs[m].mean():.1f} km average leg" if name == "distance" else VALUE_TEXT.get(name, name)
        parts.append(f"{name} ({what}) {'adds' if g[j] >= 0 else 'saves'} {abs(g[j]):.1f} min per leg")
    stops = " -> ".join(["Depot"] + [p.names[x] for x in r] + ["Depot"])
    lines.append(f"Bike {k + 1}: {stops}")
    lines.append(f"   The model predicts about {pred_l[m].mean():.1f} min per leg. Main reasons: "
                 + "; ".join(parts) + ".")
    lines.append("")
lines.append("Note: SHAP explains the travel-time MODEL. If Phase 3 used the HYBRID method, "
             "the time table uses the model only for the traffic/weather/hour slowdown factor.")
text = "\n".join(lines)
(C.RESULTS_DIR / "route_explanations.txt").write_text(text, encoding="utf-8")
print("\n=== Plain-English explanation per route ===")
print(text)
print("\nSaved: results/shap_global.png, shap_scenario.png, shap_example_trip.png, route_explanations.txt")
