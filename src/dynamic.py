"""Phase 8 - Dynamic re-routing (the real-world part).

We take the final plan (GA + SA) and simulate two disruptions during the day:

  EVENT 1 - a NEW ORDER arrives at minute T_NEW_ORDER.
            We insert it at the cheapest position among the stops NOT yet visited.
  EVENT 2 - a ROAD gets BLOCKED at minute T_ROAD_BLOCK.
            We make that road BLOCK_FACTOR times slower and re-optimize only
            the stops not yet visited.

For each event we compare:
  - quick fix methods (cheapest insertion, or a short Simulated Annealing), and
  - starting from scratch (GA + SA from random, for the same remaining stops).
We measure how fast each is and how good the new plan is.

Simplification (write it in the report): a bike that is on the road between two
stops is treated as standing at the last stop it finished.

Run from the project folder (after Phase 6):
    py src/dynamic.py
"""
import json
import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import routing_core as R

# ----------------------------- settings -------------------------------
T_NEW_ORDER = 15        # minute of the day when the new order arrives
T_ROAD_BLOCK = 30       # minute when a road gets blocked
BLOCK_FACTOR = 5.0      # blocked road becomes this many times slower
WARM_SA_STEPS = 8000    # short polish after a disruption
COLD_GA_POP = 100       # starting from scratch: GA settings
COLD_GA_GENS = 200
COLD_SA_STEPS = 20000
T_START, T_END = 10.0, 0.01
# ----------------------------------------------------------------------

p0 = R.load_problem()
K = p0.n_vehicles
random.seed(C.SEED)
rng = np.random.default_rng(C.SEED)

routes_file = C.RESULTS_DIR / "sa_routes.json"
if not routes_file.exists():
    routes_file = C.RESULTS_DIR / "ga_routes.json"
if not routes_file.exists():
    raise SystemExit("No routes found. Run Phase 5 and 6 first.")
routes0 = json.load(open(routes_file))["routes"]
print(f"Starting plan: {routes_file.name}")


# ---------- helpers ----------
def split_routes(routes, p, t_now):
    """Split every route into (already visited, still to visit) at time t_now."""
    pres, sufs = [], []
    for r in routes:
        t, prev, done = 0.0, 0, 0
        for c in r:
            finish = t + p.T[prev, c] + R.SERVICE_MIN
            if finish <= t_now:
                t, prev, done = finish, c, done + 1
            else:
                break
        pres.append(list(r[:done]))
        sufs.append(list(r[done:]))
    return pres, sufs


def full(pres, sufs):
    return [pre + suf for pre, suf in zip(pres, sufs)]


def metrics(pres, sufs, p):
    return R.evaluate(full(pres, sufs), p)


def cost(pres, sufs, p):
    return metrics(pres, sufs, p)["cost"]


def neighbour_suffix(sufs):
    """Small change that only touches stops not yet visited."""
    new = [list(s) for s in sufs]
    nonempty = [i for i in range(len(new)) if new[i]]
    if not nonempty:
        return new
    m = random.random()
    if m < 0.35:                                   # swap two stops
        a, b = random.choice(nonempty), random.choice(nonempty)
        i, j = random.randrange(len(new[a])), random.randrange(len(new[b]))
        new[a][i], new[b][j] = new[b][j], new[a][i]
    elif m < 0.75:                                 # move one stop
        a = random.choice(nonempty)
        c = new[a].pop(random.randrange(len(new[a])))
        b = random.randrange(len(new))
        new[b].insert(random.randint(0, len(new[b])), c)
    else:                                          # reverse a piece of one route
        longer = [i for i in nonempty if len(new[i]) >= 2]
        if longer:
            a = random.choice(longer)
            i, j = sorted(random.sample(range(len(new[a])), 2))
            new[a][i:j + 1] = reversed(new[a][i:j + 1])
    return new


def sa_suffix(pres, sufs, p, steps):
    cur, cur_c = sufs, cost(pres, sufs, p)
    best, best_c = [list(s) for s in cur], cur_c
    alpha = (T_END / T_START) ** (1.0 / steps)
    T = T_START
    for _ in range(steps):
        cand = neighbour_suffix(cur)
        cc = cost(pres, cand, p)
        d = cc - cur_c
        if d < 0 or random.random() < math.exp(-d / T):
            cur, cur_c = cand, cc
            if cur_c < best_c:
                best, best_c = [list(s) for s in cur], cur_c
        T *= alpha
    return best


def cheapest_insertion(pres, sufs, p, new_stop):
    """Try the new stop at every possible place (not yet visited) and keep the cheapest."""
    best, best_c = None, float("inf")
    for k in range(len(sufs)):
        for pos in range(len(sufs[k]) + 1):
            trial = [list(s) for s in sufs]
            trial[k].insert(pos, new_stop)
            c = cost(pres, trial, p)
            if c < best_c:
                best, best_c = trial, c
    return best


def cold_start(pres, remaining, p):
    """Start from scratch: GA then SA over the remaining stops only."""
    m = len(remaining)
    if m == 0:
        return [[] for _ in range(K)]
    L = m + K - 1

    def decode(ch):
        sufs, cur = [], []
        for g in ch:
            if g >= m:
                sufs.append(cur)
                cur = []
            else:
                cur.append(remaining[g])
        sufs.append(cur)
        return sufs

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
    for _ in range(COLD_GA_POP):
        g = list(range(L))
        random.shuffle(g)
        pop.append(g)
    costs = [cost(pres, decode(c), p) for c in pop]
    for _ in range(COLD_GA_GENS):
        order = np.argsort(costs)
        pop = [pop[i] for i in order]
        costs = [costs[i] for i in order]
        new_pop = pop[:2]
        while len(new_pop) < COLD_GA_POP:
            t1 = min(random.sample(range(COLD_GA_POP), 3), key=lambda i: costs[i])
            t2 = min(random.sample(range(COLD_GA_POP), 3), key=lambda i: costs[i])
            child = ox(pop[t1], pop[t2]) if random.random() < 0.9 else pop[t1][:]
            if random.random() < 0.25:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        costs = [cost(pres, decode(c), p) for c in pop]
    best = decode(pop[int(np.argmin(costs))])
    return sa_suffix(pres, best, p, COLD_SA_STEPS)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2 +
         np.cos(p1) * np.cos(p2) * np.sin((np.radians(lon2) - np.radians(lon1)) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


rows = []


def add_row(event, method, seconds, m):
    rows.append({"event": event, "method": method, "seconds": round(seconds, 3),
                 "cost": m["cost"], "total_time_min": m["total_time_min"],
                 "total_km": m["total_km"], "imbalance_min": m["imbalance_min"],
                 "overload": m["overload"]})
    print(f"  {method:34s} | {seconds:7.3f} s | cost {m['cost']:8.2f} | "
          f"time {m['total_time_min']:7.2f} min | imbalance {m['imbalance_min']:5.1f} | "
          f"overload {m['overload']}")


# =====================================================================
# EVENT 1: a new order arrives
# =====================================================================
print(f"\n=== EVENT 1: new order arrives at minute {T_NEW_ORDER} ===")
pres, sufs = split_routes(routes0, p0, T_NEW_ORDER)
left = sum(len(s) for s in sufs)
print(f"Already visited: {sum(len(x) for x in pres)} stops | still to visit: {left} stops")

# new customer: a random place inside the area of the existing customers
new_lat = float(rng.uniform(p0.lat[1:].min(), p0.lat[1:].max()))
new_lon = float(rng.uniform(p0.lon[1:].min(), p0.lon[1:].max()))
d_new = haversine_km(new_lat, new_lon, p0.lat, p0.lon)
off = ~np.eye(len(p0.names), dtype=bool) & (p0.D > 0)
ratio = float(np.median(p0.T[off] / p0.D[off]))          # minutes per km in the current situation
n0 = len(p0.names)
T1 = np.zeros((n0 + 1, n0 + 1)); T1[:n0, :n0] = p0.T
D1 = np.zeros((n0 + 1, n0 + 1)); D1[:n0, :n0] = p0.D
T1[n0, :n0] = T1[:n0, n0] = d_new * ratio
D1[n0, :n0] = D1[:n0, n0] = d_new
p1 = R.Problem(list(p0.names) + ["NEW ORDER"], np.append(p0.lat, new_lat),
               np.append(p0.lon, new_lon), np.append(p0.demand, 1), T1, D1,
               p0.n_vehicles, p0.capacity)
new_stop = n0
print(f"New order at ({new_lat:.4f}, {new_lon:.4f}), {d_new[0]:.1f} km from the depot.")

t0 = time.perf_counter()
sufs_ins = cheapest_insertion(pres, sufs, p1, new_stop)
t_ins = time.perf_counter() - t0
add_row("1 new order", "Cheapest insertion", t_ins, metrics(pres, sufs_ins, p1))

t0 = time.perf_counter()
sufs_warm = sa_suffix(pres, sufs_ins, p1, WARM_SA_STEPS)
t_warm = t_ins + (time.perf_counter() - t0)
add_row("1 new order", "Insertion + short SA", t_warm, metrics(pres, sufs_warm, p1))

t0 = time.perf_counter()
remaining = [c for s in sufs for c in s] + [new_stop]
sufs_cold = cold_start(pres, remaining, p1)
t_cold = time.perf_counter() - t0
add_row("1 new order", "From scratch (GA + SA)", t_cold, metrics(pres, sufs_cold, p1))
print(f"  -> Insertion is {t_cold / max(t_ins, 1e-9):.0f}x faster than starting from scratch.")

sufs_after1 = sufs_warm
routes1 = full(pres, sufs_after1)
R.plot_routes(routes1, p1, f"After new order (minute {T_NEW_ORDER})", C.RESULTS_DIR / "dynamic_event1_routes.png")

# =====================================================================
# EVENT 2: a road gets blocked
# =====================================================================
print(f"\n=== EVENT 2: road blocked at minute {T_ROAD_BLOCK} ===")
pres2, sufs2 = split_routes(routes1, p1, T_ROAD_BLOCK)
left2 = sum(len(s) for s in sufs2)
print(f"Already visited: {sum(len(x) for x in pres2)} stops | still to visit: {left2} stops")

# the blocked road = the slowest road still ahead of a bike
best_leg, best_t = None, -1.0
for pre, suf in zip(pres2, sufs2):
    if not suf:
        continue
    path = [pre[-1] if pre else 0] + suf + [0]
    for a, b in zip(path[:-1], path[1:]):
        if p1.T[a, b] > best_t:
            best_leg, best_t = (a, b), float(p1.T[a, b])
if best_leg is None:
    raise SystemExit("All stops are already visited at the blocking time. "
                     "Lower T_ROAD_BLOCK at the top of the file and run again.")
a, b = best_leg
T2 = p1.T.copy()
T2[a, b] *= BLOCK_FACTOR
T2[b, a] *= BLOCK_FACTOR
p2 = R.Problem(p1.names, p1.lat, p1.lon, p1.demand, T2, p1.D, p1.n_vehicles, p1.capacity)
print(f"Blocked road: {p1.names[a]}  <->  {p1.names[b]}  "
      f"({best_t:.1f} min becomes {T2[a, b]:.1f} min)")

add_row("2 road blocked", "Do nothing (keep old plan)", 0.0, metrics(pres2, sufs2, p2))

t0 = time.perf_counter()
sufs_w2 = sa_suffix(pres2, sufs2, p2, WARM_SA_STEPS)
t_w2 = time.perf_counter() - t0
add_row("2 road blocked", "Re-optimize remaining (short SA)", t_w2, metrics(pres2, sufs_w2, p2))

t0 = time.perf_counter()
remaining2 = [c for s in sufs2 for c in s]
sufs_c2 = cold_start(pres2, remaining2, p2)
t_c2 = time.perf_counter() - t0
add_row("2 road blocked", "From scratch (GA + SA)", t_c2, metrics(pres2, sufs_c2, p2))
print(f"  -> Short re-optimization is {t_c2 / max(t_w2, 1e-9):.1f}x faster than starting from scratch.")

routes2 = full(pres2, sufs_w2)
R.plot_routes(routes2, p2, f"After road block (minute {T_ROAD_BLOCK})", C.RESULTS_DIR / "dynamic_event2_routes.png")

# =====================================================================
# Save + summary
# =====================================================================
table = pd.DataFrame(rows).round(2)
print("\n=== Summary (lower is better) ===")
print(table.to_string(index=False))

C.RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(C.RESULTS_DIR / "phase8_dynamic.csv", index=False)
with open(C.RESULTS_DIR / "dynamic_routes.json", "w") as f:
    json.dump({"after_new_order": routes1, "after_road_block": routes2,
               "new_order_location": [new_lat, new_lon],
               "blocked_road": [p1.names[a], p1.names[b]]}, f)

plt.figure(figsize=(7, 4))
labels = ["New order", "Road blocked"]
quick = [t_warm, t_w2]
scratch = [t_cold, t_c2]
x = np.arange(2)
plt.bar(x - 0.2, quick, 0.4, label="Quick fix (warm start)")
plt.bar(x + 0.2, scratch, 0.4, label="From scratch (GA + SA)")
plt.xticks(x, labels)
plt.yscale("log")
plt.ylabel("Time to re-plan (seconds, log scale)")
plt.title("How fast is re-planning?")
plt.legend(); plt.grid(True, axis="y"); plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "dynamic_speed.png", dpi=150)
plt.close()

print("\nSaved: results/phase8_dynamic.csv, dynamic_routes.json, dynamic_speed.png, "
      "dynamic_event1_routes.png, dynamic_event2_routes.png")
print("\nPhase 8 done.")
