"""Shared building blocks for the route optimizer (used in Phases 4 to 8).

Nothing runs when you open this file. Other scripts import it.
It contains:
  - the settings/assumptions of the delivery problem,
  - load_problem(): reads the Ajmer stops and the time/distance tables from Phase 3,
  - evaluate(): gives any set of routes a score (the 'cost function'),
  - nearest_neighbour() and random_feasible(): simple ways to build routes,
  - plot_routes(): draws routes on a picture.
"""
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

# ----------------------------------------------------------------------
# Settings. These are ASSUMPTIONS: write them in the report.
# ----------------------------------------------------------------------
N_VEHICLES = 3            # number of delivery bikes
CAPACITY = None           # max load per bike. None = chosen automatically (30% spare room)
SERVICE_MIN = 5           # minutes spent at each customer (handing over the order)
CIRCUITY = 1.3            # road distance = straight-line distance x 1.3 (roads are not straight)
CO2_KG_PER_KM = 0.05      # ASSUMED placeholder for a motorcycle; we will cite a source in Phase 7
W_BALANCE = 0.5           # how much we dislike one rider working much longer than another
W_CO2 = 0.0               # how much we care about CO2 (0 = only time; used for eco-routing later)
OVERLOAD_PENALTY = 1000.0 # huge punishment for carrying more than the capacity

PROCESSED = C.ROOT / "data" / "processed"
STOPS_FILE = PROCESSED / "ajmer_stops_clean.csv"
TIME_FILE = PROCESSED / "time_matrix.csv"
DIST_FILE = PROCESSED / "distance_matrix.csv"


@dataclass
class Problem:
    names: list
    lat: np.ndarray
    lon: np.ndarray
    demand: np.ndarray
    T: np.ndarray            # travel minutes between stops
    D: np.ndarray            # straight-line km between stops
    n_vehicles: int
    capacity: int

    @property
    def n_customers(self):
        return len(self.names) - 1   # stop 0 is the depot


def load_problem():
    for f in (STOPS_FILE, TIME_FILE, DIST_FILE):
        if not f.exists():
            raise SystemExit(f"{f.name} not found. Run Phase 3 first:  py src/build_matrix.py")
    stops = pd.read_csv(STOPS_FILE)
    T = pd.read_csv(TIME_FILE, index_col=0).to_numpy(dtype=float)
    D = pd.read_csv(DIST_FILE, index_col=0).to_numpy(dtype=float)
    n = len(stops)
    if T.shape != (n, n) or D.shape != (n, n):
        raise SystemExit("The tables do not match the stops file. Run Phase 3 again.")
    demand = stops["demand"].to_numpy(dtype=int)
    cap = CAPACITY
    if cap is None:
        cap = max(int(math.ceil(1.3 * demand.sum() / N_VEHICLES)), int(demand.max()))
    return Problem(list(stops["name"]), stops["lat"].to_numpy(), stops["lon"].to_numpy(),
                   demand, T, D, N_VEHICLES, int(cap))


def evaluate(routes, p):
    """Score a solution. 'routes' is a list with one list of customer numbers per bike.
    Every bike starts and ends at the depot (stop 0). LOWER cost is BETTER."""
    times, kms, loads = [], [], []
    for r in routes:
        if len(r) == 0:
            times.append(0.0); kms.append(0.0); loads.append(0)
            continue
        path = [0] + list(r) + [0]
        a, b = path[:-1], path[1:]
        times.append(float(p.T[a, b].sum()) + SERVICE_MIN * len(r))
        kms.append(float(p.D[a, b].sum()))
        loads.append(int(p.demand[list(r)].sum()))
    overload = sum(max(0, ld - p.capacity) for ld in loads)
    total_time = sum(times)
    total_km = sum(kms)
    co2 = total_km * CIRCUITY * CO2_KG_PER_KM
    imbalance = max(times) - min(times)
    cost = total_time + W_BALANCE * imbalance + W_CO2 * co2 + OVERLOAD_PENALTY * overload
    return {"cost": cost, "total_time_min": total_time, "total_km": total_km,
            "co2_kg": co2, "longest_route_min": max(times), "imbalance_min": imbalance,
            "overload": overload, "route_times": times, "route_kms": kms, "loads": loads}


def nearest_neighbour(p):
    """Simple baseline: each bike always drives to the closest customer it can still carry."""
    unvisited = set(range(1, len(p.names)))
    routes = []
    for _ in range(p.n_vehicles):
        route, load, cur = [], 0, 0
        while True:
            fits = [j for j in unvisited if load + p.demand[j] <= p.capacity]
            if not fits:
                break
            nxt = min(fits, key=lambda j: p.T[cur, j])
            route.append(nxt); unvisited.remove(nxt)
            load += int(p.demand[nxt]); cur = nxt
        routes.append(route)
    if unvisited:   # could not fit: give the rest to the last bike (it will be punished)
        routes[-1].extend(sorted(unvisited, key=lambda j: p.T[0, j]))
    return routes


def random_feasible(p, rng):
    """Random baseline: customers in random order, filling one bike after another."""
    order = list(rng.permutation(np.arange(1, len(p.names))))
    routes = [[] for _ in range(p.n_vehicles)]
    loads = [0] * p.n_vehicles
    v = 0
    for j in order:
        while v < p.n_vehicles - 1 and loads[v] + p.demand[j] > p.capacity:
            v += 1
        routes[v].append(int(j)); loads[v] += int(p.demand[j])
    return routes


def plot_routes(routes, p, title, path):
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    plt.figure(figsize=(7, 7))
    for k, r in enumerate(routes):
        if not r:
            continue
        pts = [0] + list(r) + [0]
        plt.plot(p.lon[pts], p.lat[pts], "-o", color=colors[k % len(colors)],
                 label=f"Bike {k + 1}", markersize=5)
    for i in range(1, len(p.names)):
        plt.annotate(str(i), (p.lon[i], p.lat[i]), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")
    plt.scatter(p.lon[0], p.lat[0], c="red", marker="*", s=250, zorder=5, label="Depot")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
