"""Phase 6 - Simulated Annealing (the polisher).

Idea: start from the best routes found by the GA and make SMALL changes:
  - swap two customers (can be on different bikes),
  - move one customer to another position or another bike,
  - reverse a small part of one route (a '2-opt' style fix).
A better route is always kept. A worse route is sometimes kept, with a
probability exp(-extra_cost / temperature). At the start the temperature is
high (we accept many bad moves, so we do not get stuck). It cools down slowly,
so at the end we only accept improvements.

Run from the project folder (after Phase 5):
    py src/sa.py
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
T_START = 20.0        # starting temperature (cost units = minutes)
T_END = 0.01          # final temperature
STEPS = 40000         # number of small changes tried
# ----------------------------------------------------------------------

p = R.load_problem()
random.seed(C.SEED)
np.random.seed(C.SEED)

ga_file = C.RESULTS_DIR / "ga_routes.json"
if not ga_file.exists():
    raise SystemExit("results/ga_routes.json not found. Run Phase 5 first:  py src/ga.py")
ga_routes = json.load(open(ga_file))["routes"]
assert sorted(c for r in ga_routes for c in r) == list(range(1, p.n_customers + 1)), \
    "ga_routes.json does not match the current problem. Run Phase 5 again."


def cost(routes):
    return R.evaluate(routes, p)["cost"]


def neighbour(routes):
    """Return a slightly changed copy of the routes."""
    new = [list(r) for r in routes]
    k = len(new)
    move = random.random()
    nonempty = [i for i in range(k) if new[i]]

    if move < 0.35:                               # swap two customers
        a, b = random.choice(nonempty), random.choice(nonempty)
        i, j = random.randrange(len(new[a])), random.randrange(len(new[b]))
        new[a][i], new[b][j] = new[b][j], new[a][i]
    elif move < 0.75:                             # move one customer
        a = random.choice(nonempty)
        c = new[a].pop(random.randrange(len(new[a])))
        b = random.randrange(k)
        new[b].insert(random.randint(0, len(new[b])), c)
    else:                                         # reverse a segment in one route
        longer = [i for i in range(k) if len(new[i]) >= 2]
        if longer:
            a = random.choice(longer)
            i, j = sorted(random.sample(range(len(new[a])), 2))
            new[a][i:j + 1] = reversed(new[a][i:j + 1])
    return new


def anneal(start):
    cur, cur_cost = start, cost(start)
    best, best_cost = [list(r) for r in cur], cur_cost
    alpha = (T_END / T_START) ** (1.0 / STEPS)    # cooling factor per step
    T = T_START
    cur_hist, best_hist = [cur_cost], [best_cost]
    accepted_worse = 0

    for step in range(STEPS):
        cand = neighbour(cur)
        cand_cost = cost(cand)
        delta = cand_cost - cur_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            if delta > 0:
                accepted_worse += 1
            cur, cur_cost = cand, cand_cost
            if cur_cost < best_cost:
                best, best_cost = [list(r) for r in cur], cur_cost
        T *= alpha
        cur_hist.append(cur_cost)
        best_hist.append(best_cost)
        if step % 5000 == 0 or step == STEPS - 1:
            print(f"Step {step:5d} | temp {T:7.3f} | current {cur_cost:8.2f} | best {best_cost:8.2f}")
    print(f"Worse moves accepted (to escape local optima): {accepted_worse}")
    return best, cur_hist, best_hist


start_time = time.time()
sa_routes, cur_hist, best_hist = anneal(ga_routes)
elapsed = time.time() - start_time

nn = R.evaluate(R.nearest_neighbour(p), p)
ga = R.evaluate(ga_routes, p)
sa = R.evaluate(sa_routes, p)

print("\n=== GA + SA routes ===")
for k, r in enumerate(sa_routes, 1):
    path = " -> ".join(["Depot"] + [p.names[i] for i in r] + ["Depot"])
    print(f"Bike {k}: {path}")
    print(f"        load {sa['loads'][k-1]}/{p.capacity} | "
          f"time {sa['route_times'][k-1]:.0f} min | distance {sa['route_kms'][k-1]:.1f} km")

cols = ["cost", "total_time_min", "total_km", "co2_kg",
        "longest_route_min", "imbalance_min", "overload"]
table = pd.DataFrame([{"method": "Nearest neighbour", **{k: nn[k] for k in cols}},
                      {"method": "Genetic Algorithm", **{k: ga[k] for k in cols}},
                      {"method": "GA + Simulated Annealing", **{k: sa[k] for k in cols}}]).round(2)
print("\n=== Scores (lower is better) ===")
print(table.to_string(index=False))

sa_gain = (ga["cost"] - sa["cost"]) / ga["cost"] * 100
total_gain = (nn["cost"] - sa["cost"]) / nn["cost"] * 100
print(f"\nSA added {sa_gain:.2f}% improvement on top of the GA.   <-- use this in your report")
print(f"GA + SA is {total_gain:.1f}% better than nearest neighbour overall.")
print(f"SA run time: {elapsed:.1f} seconds")

C.RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(C.RESULTS_DIR / "phase6_sa.csv", index=False)
with open(C.RESULTS_DIR / "sa_routes.json", "w") as f:
    json.dump({"routes": sa_routes, "cost": float(sa["cost"]),
               "ga_cost": float(ga["cost"]), "sa_gain_percent": float(sa_gain),
               "seconds": elapsed}, f)

plt.figure(figsize=(7, 4))
plt.plot(cur_hist, color="lightgray", label="Current cost (goes up and down)")
plt.plot(best_hist, color="tab:blue", label="Best cost so far")
plt.axhline(ga["cost"], color="green", linestyle="--", label="GA result")
plt.xlabel("Step"); plt.ylabel("Cost (lower is better)")
plt.title("Simulated Annealing")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "sa_convergence.png", dpi=150)
plt.close()

R.plot_routes(sa_routes, p, "GA + Simulated Annealing routes (Ajmer)", C.RESULTS_DIR / "sa_routes.png")
print("Saved: results/phase6_sa.csv, sa_routes.json, sa_convergence.png, sa_routes.png")
print("\nPhase 6 done.")
