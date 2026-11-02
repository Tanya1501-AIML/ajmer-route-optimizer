"""Phase 9 - Dashboard (the demo for your evaluators).

Run from the project folder:
    pip install streamlit folium
    streamlit run src/app.py

What you can do on the page:
  - choose the number of bikes, the hour, the weather and the traffic (left side),
  - press 'Plan routes' to build routes with GA + SA,
  - see a map with one colour per bike, a metrics table, and a SHAP explanation,
  - press 'Add new order' or 'Block a road' to see live re-routing.
"""
import json
import math
import random
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import folium
import xgboost as xgb

import config as C
import routing_core as R
from features import make_features, WEATHER_VALUES

# ------------------------------------------------------------------
# KEEP THESE SAME AS IN build_matrix.py
# ------------------------------------------------------------------
WEEKDAY = 4
VEHICLE = "motorcycle"
MULTIPLE_DELIVERIES = 1
FESTIVAL = "No"
BASE_SPEED_KMPH = 25
BLOCK_FACTOR = 5.0
# starting colours; you can change every one of them live in the sidebar
DEFAULT_BIKE_COLORS = ["#2a81cb", "#2aad27", "#cb8427", "#9c2bcb", "#a23336"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FEATURES_FILE = C.ROOT / "models" / "eta_features.json"

st.set_page_config(page_title="Ajmer Delivery Route Optimizer", layout="wide")


# ======================================================================
# Loading things once
# ======================================================================
@st.cache_resource
def load_everything():
    model = xgb.XGBRegressor()
    model.load_model(str(C.MODEL_FILE))
    feature_names = json.loads(FEATURES_FILE.read_text())
    stops = pd.read_csv(R.STOPS_FILE)
    D = pd.read_csv(R.DIST_FILE, index_col=0).to_numpy(dtype=float)
    return model, feature_names, stops, D


model, FN, stops, D0 = load_everything()


# ======================================================================
# Travel time from the model (same idea as Phase 3)
# ======================================================================
def make_df(dists, hour, weather, traffic):
    d = pd.DataFrame({"distance_km": np.asarray(dists, dtype=float)})
    d["hour"], d["weekday"] = hour, WEEKDAY
    d["is_weekend"] = int(WEEKDAY >= 5)
    d["traffic"], d["weather"], d["vehicle"] = traffic, weather, VEHICLE
    d["multiple_deliveries"], d["festival"] = MULTIPLE_DELIVERIES, FESTIVAL
    return d


def predict_minutes(dists, hour, weather, traffic):
    return model.predict(make_features(make_df(dists, hour, weather, traffic))[FN])


def build_time_matrix(D, hour, weather, traffic):
    extra = float(predict_minutes([12], hour, weather, traffic)[0] -
                  predict_minutes([2], hour, weather, traffic)[0])
    speed = 10 / (extra / 60) if extra > 0 else float("inf")
    if speed > 45:        # distance effect too weak -> hybrid method
        ref = predict_minutes([8], 11, "Sunny", "Low")[0]
        now = predict_minutes([8], hour, weather, traffic)[0]
        T = D / BASE_SPEED_KMPH * 60 * (now / ref)
    else:
        T = predict_minutes(D.ravel(), hour, weather, traffic).reshape(D.shape)
    T = np.asarray(T, dtype=float)
    np.fill_diagonal(T, 0)
    return T


# ======================================================================
# Optimizer pieces (same ideas as ga.py, sa.py, dynamic.py, but compact)
# ======================================================================
def split_routes(routes, p, t_now):
    pres, sufs = [], []
    for r in routes:
        t, prev, done = 0.0, 0, 0
        for c in r:
            fin = t + p.T[prev, c] + R.SERVICE_MIN
            if fin <= t_now:
                t, prev, done = fin, c, done + 1
            else:
                break
        pres.append(list(r[:done]))
        sufs.append(list(r[done:]))
    return pres, sufs


def full(pres, sufs):
    return [a + b for a, b in zip(pres, sufs)]


def cost(pres, sufs, p):
    return R.evaluate(full(pres, sufs), p)["cost"]


def neighbour(sufs):
    new = [list(s) for s in sufs]
    ne = [i for i in range(len(new)) if new[i]]
    if not ne:
        return new
    m = random.random()
    if m < 0.35:
        a, b = random.choice(ne), random.choice(ne)
        i, j = random.randrange(len(new[a])), random.randrange(len(new[b]))
        new[a][i], new[b][j] = new[b][j], new[a][i]
    elif m < 0.75:
        a = random.choice(ne)
        c = new[a].pop(random.randrange(len(new[a])))
        b = random.randrange(len(new))
        new[b].insert(random.randint(0, len(new[b])), c)
    else:
        longer = [i for i in ne if len(new[i]) >= 2]
        if longer:
            a = random.choice(longer)
            i, j = sorted(random.sample(range(len(new[a])), 2))
            new[a][i:j + 1] = reversed(new[a][i:j + 1])
    return new


def sa(pres, sufs, p, steps, t_start=10.0, t_end=0.01):
    cur, cur_c = sufs, cost(pres, sufs, p)
    best, best_c = [list(s) for s in cur], cur_c
    alpha = (t_end / t_start) ** (1.0 / max(steps, 1))
    T = t_start
    for _ in range(steps):
        cand = neighbour(cur)
        cc = cost(pres, cand, p)
        d = cc - cur_c
        if d < 0 or random.random() < math.exp(-d / T):
            cur, cur_c = cand, cc
            if cur_c < best_c:
                best, best_c = [list(s) for s in cur], cur_c
        T *= alpha
    return best


def cheapest_insertion(pres, sufs, p, new_stop):
    best, best_c = None, float("inf")
    for k in range(len(sufs)):
        for pos in range(len(sufs[k]) + 1):
            trial = [list(s) for s in sufs]
            trial[k].insert(pos, new_stop)
            c = cost(pres, trial, p)
            if c < best_c:
                best, best_c = trial, c
    return best


def plan_from_scratch(p, pop_size=60, gens=100, sa_steps=6000):
    """GA then SA over all customers (nothing visited yet)."""
    n, k = p.n_customers, p.n_vehicles
    L = n + k - 1
    pres = [[] for _ in range(k)]

    def decode(ch):
        out, cur = [], []
        for g in ch:
            if g > n:
                out.append(cur)
                cur = []
            else:
                cur.append(int(g))
        out.append(cur)
        return out

    def mutate(ch):
        c = ch[:]
        i, j = sorted(random.sample(range(L), 2)) if L >= 2 else (0, 0)
        r = random.random()
        if r < 0.4:
            c[i], c[j] = c[j], c[i]
        elif r < 0.8:
            c[i:j + 1] = reversed(c[i:j + 1])
        else:
            g = c.pop(i)
            c.insert(j, g)
        return c

    def ox(p1, p2):
        if L < 2:
            return p1[:]
        a, b = sorted(random.sample(range(L), 2))
        child = [None] * L
        child[a:b + 1] = p1[a:b + 1]
        used = set(child[a:b + 1])
        fill = iter(g for g in p2 if g not in used)
        for i in range(L):
            if child[i] is None:
                child[i] = next(fill)
        return child

    pop = []
    for _ in range(pop_size):
        g = list(range(1, n + k))
        random.shuffle(g)
        pop.append(g)
    costs = [cost(pres, decode(c), p) for c in pop]
    for _ in range(gens):
        order = np.argsort(costs)
        pop = [pop[i] for i in order]
        costs = [costs[i] for i in order]
        new_pop = pop[:2]
        while len(new_pop) < pop_size:
            t1 = min(random.sample(range(pop_size), 3), key=lambda i: costs[i])
            t2 = min(random.sample(range(pop_size), 3), key=lambda i: costs[i])
            child = ox(pop[t1], pop[t2]) if random.random() < 0.9 else pop[t1][:]
            if random.random() < 0.25:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        costs = [cost(pres, decode(c), p) for c in pop]
    best = decode(pop[int(np.argmin(costs))])
    return sa(pres, best, p, sa_steps)


def make_problem(n_veh, T, D):
    demand = stops["demand"].to_numpy(dtype=int)
    cap = max(int(math.ceil(1.3 * demand.sum() / n_veh)), int(demand.max()))
    return R.Problem(list(stops["name"]), stops["lat"].to_numpy(), stops["lon"].to_numpy(),
                     demand, T, D, n_veh, cap)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2 +
         np.cos(p1) * np.cos(p2) * np.sin((np.radians(lon2) - np.radians(lon1)) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


# ======================================================================
# The three actions
# ======================================================================
def do_plan(settings):
    random.seed(C.SEED)
    T = build_time_matrix(D0, settings["hour"], settings["weather"], settings["traffic"])
    p = make_problem(settings["n_veh"], T, D0.copy())
    t0 = time.perf_counter()
    routes = plan_from_scratch(p)
    sec = time.perf_counter() - t0
    st.session_state.state = {
        "p": p, "routes": routes, "settings": dict(settings), "blocked": [],
        "new_orders": [], "log": [f"Planned {p.n_customers} customers for {p.n_vehicles} bikes "
                                  f"in {sec:.1f} s (GA + SA)."]}


def do_add_order(minute):
    s = st.session_state.state
    p, routes = s["p"], s["routes"]
    rng = np.random.default_rng(len(s["new_orders"]) + C.SEED)
    lat = float(rng.uniform(p.lat[1:].min(), p.lat[1:].max()))
    lon = float(rng.uniform(p.lon[1:].min(), p.lon[1:].max()))
    d_new = haversine_km(lat, lon, p.lat, p.lon)
    off = ~np.eye(len(p.names), dtype=bool) & (p.D > 0)
    ratio = float(np.median(p.T[off] / p.D[off]))
    n0 = len(p.names)
    T1 = np.zeros((n0 + 1, n0 + 1)); T1[:n0, :n0] = p.T
    D1 = np.zeros((n0 + 1, n0 + 1)); D1[:n0, :n0] = p.D
    T1[n0, :n0] = T1[:n0, n0] = d_new * ratio
    D1[n0, :n0] = D1[:n0, n0] = d_new
    name = f"NEW ORDER {len(s['new_orders']) + 1}"
    p1 = R.Problem(list(p.names) + [name], np.append(p.lat, lat), np.append(p.lon, lon),
                   np.append(p.demand, 1), T1, D1, p.n_vehicles, p.capacity)
    random.seed(C.SEED)
    t0 = time.perf_counter()
    pres, sufs = split_routes(routes, p, minute)
    sufs = cheapest_insertion(pres, sufs, p1, n0)
    sufs = sa(pres, sufs, p1, 3000)
    sec = time.perf_counter() - t0
    bike = next(k for k, sf in enumerate(sufs) if n0 in sf) + 1
    s["p"], s["routes"] = p1, full(pres, sufs)
    s["new_orders"].append(n0)
    s["log"].append(f"{name} arrived at minute {minute}: given to Bike {bike}, "
                    f"re-planned only the unvisited stops in {sec * 1000:.0f} ms.")


def do_block_road(minute):
    s = st.session_state.state
    p, routes = s["p"], s["routes"]
    pres, sufs = split_routes(routes, p, minute)
    best, best_t = None, -1.0
    for pre, suf in zip(pres, sufs):
        if not suf:
            continue
        path = [pre[-1] if pre else 0] + suf + [0]
        for a, b in zip(path[:-1], path[1:]):
            if p.T[a, b] > best_t:
                best, best_t = (a, b), float(p.T[a, b])
    if best is None:
        s["log"].append("Nothing to block: every stop is already visited at this time.")
        return
    a, b = best
    T2 = p.T.copy()
    T2[a, b] *= BLOCK_FACTOR
    T2[b, a] *= BLOCK_FACTOR
    p2 = R.Problem(p.names, p.lat, p.lon, p.demand, T2, p.D, p.n_vehicles, p.capacity)
    random.seed(C.SEED)
    t0 = time.perf_counter()
    sufs = sa(pres, sufs, p2, 6000)
    sec = time.perf_counter() - t0
    s["p"], s["routes"] = p2, full(pres, sufs)
    s["blocked"].append((a, b))
    s["log"].append(f"Road {p.names[a]} <-> {p.names[b]} blocked at minute {minute} "
                    f"({best_t:.0f} -> {best_t * BLOCK_FACTOR:.0f} min). "
                    f"Re-optimized unvisited stops in {sec * 1000:.0f} ms.")


# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.title("Settings")
n_veh = st.sidebar.slider("Number of bikes", 1, 5, int(R.N_VEHICLES), key="n_veh")
hour = st.sidebar.slider("Hour of the day", 0, 23, 18, key="hour")
weather = st.sidebar.selectbox("Weather", WEATHER_VALUES, key="weather")
traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High", "Jam"], index=2, key="traffic")
minute = st.sidebar.slider("Minutes since the bikes left (for live events)", 0, 90, 15, key="minute")
settings = {"n_veh": n_veh, "hour": hour, "weather": weather, "traffic": traffic}
plan_btn = st.sidebar.button("Plan routes", type="primary", key="plan")
st.sidebar.caption("Changing a slider does not re-plan by itself. Press 'Plan routes'.")

with st.sidebar.expander("Colours (change anytime, no re-planning needed)", expanded=True):
    bike_colors = [st.color_picker(f"Bike {k + 1} route", DEFAULT_BIKE_COLORS[k], key=f"col_bike_{k}")
                   for k in range(5)]
    depot_color = st.color_picker("Depot", "#d33b3b", key="col_depot")
    new_color = st.color_picker("New order", "#f2b705", key="col_new")
    block_color = st.color_picker("Blocked road", "#e00000", key="col_block")

st.title("Ajmer Delivery Route Optimizer")
st.caption("Travel times from an XGBoost model, routes from a Genetic Algorithm + Simulated Annealing, "
           "explained with SHAP, with live re-routing.")

if plan_btn or "state" not in st.session_state:
    with st.spinner("Planning routes..."):
        do_plan(settings)

state = st.session_state.state
if state["settings"] != settings:
    st.info("Settings changed. Press 'Plan routes' to apply them.")

c1, c2 = st.columns(2)
if c1.button("Add new order", key="add"):
    do_add_order(minute)
if c2.button("Block a road", key="block"):
    do_block_road(minute)

p, routes = state["p"], state["routes"]
S = state["settings"]
m = R.evaluate(routes, p)
nn = R.evaluate(R.nearest_neighbour(p), p)

# ======================================================================
# Metrics
# ======================================================================
st.subheader(f"Plan for {DAYS[WEEKDAY]} {S['hour']}:00, {S['weather']}, {S['traffic']} traffic, {S['n_veh']} bikes")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total time (min)", f"{m['total_time_min']:.1f}", f"{m['total_time_min'] - nn['total_time_min']:.1f} vs nearest neighbour", delta_color="inverse")
k2.metric("Distance (km)", f"{m['total_km']:.1f}", f"{m['total_km'] - nn['total_km']:.1f}", delta_color="inverse")
k3.metric("CO2 (kg, estimate)", f"{m['co2_kg']:.2f}", f"{m['co2_kg'] - nn['co2_kg']:.2f}", delta_color="inverse")
k4.metric("Imbalance (min)", f"{m['imbalance_min']:.1f}", f"{m['imbalance_min'] - nn['imbalance_min']:.1f}", delta_color="inverse")
if m["overload"] > 0:
    st.warning("A bike is overloaded in this plan.")

# ======================================================================
# Map
# ======================================================================
pres_now, _ = split_routes(routes, p, minute)
fmap = folium.Map(location=[float(p.lat.mean()), float(p.lon.mean())], zoom_start=13)
for k, r in enumerate(routes):
    if not r:
        continue
    col = bike_colors[k % len(bike_colors)]
    path = [0] + list(r) + [0]
    folium.PolyLine([(float(p.lat[i]), float(p.lon[i])) for i in path], color=col,
                    weight=4, opacity=0.85, tooltip=f"Bike {k + 1}").add_to(fmap)
    for pos, i in enumerate(r, 1):
        new = i in state["new_orders"]
        folium.CircleMarker((float(p.lat[i]), float(p.lon[i])), radius=8 if new else 6,
                            color="black" if new else col, fill=True,
                            fill_color=new_color if new else col, fill_opacity=0.95,
                            tooltip=f"Bike {k + 1}, stop {pos}: {p.names[i]}"
                                    f"{' (visited)' if pos <= len(pres_now[k]) and minute > 0 else ''}").add_to(fmap)
folium.Marker((float(p.lat[0]), float(p.lon[0])), tooltip="Depot",
              icon=folium.DivIcon(icon_size=(30, 30), icon_anchor=(15, 15),
                                  html=f'<div style="font-size:30px;line-height:30px;'
                                       f'color:{depot_color};text-shadow:0 0 3px #fff;">&#9733;</div>')
              ).add_to(fmap)
for a, b in state["blocked"]:
    folium.PolyLine([(float(p.lat[a]), float(p.lon[a])), (float(p.lat[b]), float(p.lon[b]))],
                    color=block_color, weight=8, dash_array="8", tooltip="BLOCKED ROAD").add_to(fmap)
map_html = fmap.get_root().render()
if hasattr(st, "iframe"):                      # newer Streamlit
    st.iframe(map_html, height=520)
else:                                          # older Streamlit
    components.html(map_html, height=520)
st.caption("Each colour is one bike. Big star = depot. Bigger dot with black edge = new order. Dashed thick line = blocked road.")

# ======================================================================
# Tables
# ======================================================================
left, right = st.columns(2)
with left:
    st.subheader("Each bike")
    rows = []
    for k, r in enumerate(routes):
        rows.append({"Bike": k + 1, "Stops": len(r), "Load": f"{m['loads'][k]}/{p.capacity}",
                     "Time (min)": round(m["route_times"][k], 1),
                     "Distance (km)": round(m["route_kms"][k], 1),
                     "Route": " > ".join(p.names[i].replace(" area", "") for i in r)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
with right:
    st.subheader("Compared with the simple method")
    cmp = pd.DataFrame([
        {"Method": "Nearest neighbour", "Cost": nn["cost"], "Time": nn["total_time_min"],
         "Km": nn["total_km"], "CO2": nn["co2_kg"], "Imbalance": nn["imbalance_min"]},
        {"Method": "Our plan", "Cost": m["cost"], "Time": m["total_time_min"],
         "Km": m["total_km"], "CO2": m["co2_kg"], "Imbalance": m["imbalance_min"]}]).round(2)
    st.dataframe(cmp, hide_index=True)
    gain = (nn["cost"] - m["cost"]) / nn["cost"] * 100
    st.write(f"Our plan is **{gain:.1f}%** better on the cost score.")

# ======================================================================
# SHAP explanation (TreeSHAP from XGBoost)
# ======================================================================
st.subheader("Why are trips slow or fast right now? (SHAP)")
GROUPS = {
    "distance": ["distance_km"], "hour of day": ["hour_sin", "hour_cos"],
    "traffic": ["traffic_level"],
    "weather": [c for c in FN if c.startswith("weather_")],
    "vehicle type": [c for c in FN if c.startswith("vehicle_")],
    "day of week": ["weekday", "is_weekend"],
    "extra deliveries carried": ["multiple_deliveries"], "festival": ["festival_yes"]}
GROUPS = {g: [c for c in cols if c in FN] for g, cols in GROUPS.items()}
GROUPS = {g: cols for g, cols in GROUPS.items() if cols}

legs = []
for r in routes:
    path = [0] + list(r) + [0]
    legs += [float(p.D[a, b]) for a, b in zip(path[:-1], path[1:])]
X = make_features(make_df(legs, S["hour"], S["weather"], S["traffic"]))[FN]
contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
vals, base = contrib[:, :-1], float(contrib[:, -1].mean())
idx = {c: i for i, c in enumerate(X.columns)}
G = pd.Series({g: float(vals[:, [idx[c] for c in cols]].sum(axis=1).mean())
               for g, cols in GROUPS.items()})
top = G.reindex(G.abs().sort_values(ascending=False).index)[:3]
words = "; ".join(f"{g} {'adds' if v >= 0 else 'saves'} {abs(v):.1f} min per leg" for g, v in top.items())
st.write(f"Compared with an average trip ({base:.1f} min): {words}.")
st.bar_chart(G.sort_values(), horizontal=True)
st.caption("Positive = slower than an average training trip, negative = faster.")

# ======================================================================
# Log
# ======================================================================
st.subheader("What happened")
for line in reversed(state["log"]):
    st.write("- " + line)
