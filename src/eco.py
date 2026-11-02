"""Phase 7 (part 2) - Eco-routing: fastest routes vs lowest-CO2 routes.

We run the same optimizer (GA, then SA) several times. Each time we raise the
CO2 weight (W_CO2 in routing_core.py). W_CO2 = 0 means 'only care about time'
(the fastest routes). A big W_CO2 means 'CO2 matters a lot' (eco routes).
Comparing them shows the trade-off: how much CO2 we save and how many extra
minutes it costs.

Run from the project folder:
    py src/eco.py
"""
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import routing_core as R

# ----------------------------- settings -------------------------------
CO2_WEIGHTS = [0, 20, 50, 100, 200, 400]   # 0 = fastest, bigger = greener
POP_SIZE = 100
GENERATIONS = 200
SA_STEPS = 20000
N_RESTARTS = 5                             # runs per weight (the optimizer is random)
T_START, T_END = 20.0, 0.01
# ----------------------------------------------------------------------

p = R.load_problem()
N, K = p.n_customers, p.n_vehicles
L = N + K - 1


def decode(chrom):
    routes, cur = [], []
    for g in chrom:
        if g > N:
            routes.append(cur)
            cur = []
        else:
            cur.append(int(g))
    routes.append(cur)
    return routes


def cost_of(routes):
    return R.evaluate(routes, p)["cost"]      # uses the CURRENT R.W_CO2


def mutate_chrom(c):
    c = c[:]
    i, j = sorted(random.sample(range(L), 2))
    r = random.random()
    if r < 0.4:
        c[i], c[j] = c[j], c[i]
    elif r < 0.8:
        c[i:j + 1] = reversed(c[i:j + 1])
    else:
        g = c.pop(i)
        c.insert(j, g)
    return c


def order_crossover(p1, p2):
    a, b = sorted(random.sample(range(L), 2))
    child = [None] * L
    child[a:b + 1] = p1[a:b + 1]
    used = set(child[a:b + 1])
    fill = iter(g for g in p2 if g not in used)
    for i in range(L):
        if child[i] is None:
            child[i] = next(fill)
    return child


def run_ga():
    pop = []
    for _ in range(POP_SIZE):
        g = list(range(1, N + K))
        random.shuffle(g)
        pop.append(g)
    costs = [cost_of(decode(c)) for c in pop]
    for _ in range(GENERATIONS):
        order = np.argsort(costs)
        pop = [pop[i] for i in order]
        costs = [costs[i] for i in order]
        new_pop = pop[:2]
        while len(new_pop) < POP_SIZE:
            t1 = min(random.sample(range(POP_SIZE), 3), key=lambda i: costs[i])
            t2 = min(random.sample(range(POP_SIZE), 3), key=lambda i: costs[i])
            child = order_crossover(pop[t1], pop[t2]) if random.random() < 0.9 else pop[t1][:]
            if random.random() < 0.25:
                child = mutate_chrom(child)
            new_pop.append(child)
        pop = new_pop
        costs = [cost_of(decode(c)) for c in pop]
    return decode(pop[int(np.argmin(costs))])


def neighbour(routes):
    new = [list(r) for r in routes]
    k = len(new)
    nonempty = [i for i in range(k) if new[i]]
    m = random.random()
    if m < 0.35:
        a, b = random.choice(nonempty), random.choice(nonempty)
        i, j = random.randrange(len(new[a])), random.randrange(len(new[b]))
        new[a][i], new[b][j] = new[b][j], new[a][i]
    elif m < 0.75:
        a = random.choice(nonempty)
        c = new[a].pop(random.randrange(len(new[a])))
        b = random.randrange(k)
        new[b].insert(random.randint(0, len(new[b])), c)
    else:
        longer = [i for i in range(k) if len(new[i]) >= 2]
        if longer:
            a = random.choice(longer)
            i, j = sorted(random.sample(range(len(new[a])), 2))
            new[a][i:j + 1] = reversed(new[a][i:j + 1])
    return new


def run_sa(start):
    cur, cur_c = start, cost_of(start)
    best, best_c = [list(r) for r in cur], cur_c
    alpha = (T_END / T_START) ** (1.0 / SA_STEPS)
    T = T_START
    for _ in range(SA_STEPS):
        cand = neighbour(cur)
        cc = cost_of(cand)
        d = cc - cur_c
        if d < 0 or random.random() < math.exp(-d / T):
            cur, cur_c = cand, cc
            if cur_c < best_c:
                best, best_c = [list(r) for r in cur], cur_c
        T *= alpha
    return best


# Step A: run the optimizer several times for every weight and keep ALL solutions found.
pool = []
for w in CO2_WEIGHTS:
    R.W_CO2 = w
    for s in range(N_RESTARTS):
        random.seed(C.SEED + s)
        np.random.seed(C.SEED + s)
        pool.append(run_sa(run_ga()))
    print(f"CO2 weight {w:>3}: {N_RESTARTS} runs done")

# Step B: for every weight, pick the best solution from the WHOLE pool.
# (A solution found while testing one weight may be best for another weight too.)
rows, all_routes = [], {}
for w in CO2_WEIGHTS:
    R.W_CO2 = w
    routes = min(pool, key=cost_of)
    m = R.evaluate(routes, p)
    label = "Fastest (W_CO2=0)" if w == 0 else f"W_CO2={w}"
    rows.append({"setting": label, "w_co2": w, "total_time_min": m["total_time_min"],
                 "total_km": m["total_km"], "co2_kg": m["co2_kg"],
                 "imbalance_min": m["imbalance_min"], "overload": m["overload"]})
    all_routes[label] = routes
    print(f"{label:20s} | time {m['total_time_min']:7.2f} min | "
          f"km {m['total_km']:6.2f} | CO2 {m['co2_kg']:.3f} kg | "
          f"imbalance {m['imbalance_min']:5.1f} | overload {m['overload']}")
R.W_CO2 = 0.0

table = pd.DataFrame(rows).round(3)
ok = table[table["overload"] == 0]
fast = ok.iloc[0]
eco = ok.loc[ok["co2_kg"].idxmin()]

print("\n=== Trade-off table ===")
print(table.to_string(index=False))

co2_save = (fast["co2_kg"] - eco["co2_kg"]) / fast["co2_kg"] * 100
d_time = eco["total_time_min"] - fast["total_time_min"]
d_pct = d_time / fast["total_time_min"] * 100
print("\n=== Plain-English summary (use in report) ===")
print(f"The fastest plan takes {fast['total_time_min']:.1f} min and emits {fast['co2_kg']:.3f} kg CO2.")
if eco["setting"] == fast["setting"]:
    print("No greener plan was found: the fastest plan is already the lowest-CO2 plan.")
else:
    when = (f"and takes {d_time:.1f} min ({d_pct:.1f}%) MORE total time" if d_time > 0
            else f"and also saves {abs(d_time):.1f} min ({abs(d_pct):.1f}%) of total time")
    print(f"The greenest plan ({eco['setting']}) emits {eco['co2_kg']:.3f} kg CO2, "
          f"{co2_save:.1f}% less, {when}.")
    print(f"Fairness: the gap between the longest and shortest route changes from "
          f"{fast['imbalance_min']:.1f} to {eco['imbalance_min']:.1f} min.")
print("NOTE: CO2 here = distance x a fixed assumed factor, so eco-routing really means")
print("'shortest distance'. Time and distance are strongly linked, so the trade-off is small.")

C.RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(C.RESULTS_DIR / "phase7_eco.csv", index=False)
with open(C.RESULTS_DIR / "eco_routes.json", "w") as f:
    json.dump({"fastest": all_routes[fast["setting"]], "eco": all_routes[eco["setting"]]}, f)

plt.figure(figsize=(6, 4.5))
plt.plot(table["total_time_min"], table["co2_kg"], "-o")
for _, r in table.iterrows():
    plt.annotate(str(int(r["w_co2"])), (r["total_time_min"], r["co2_kg"]),
                 xytext=(4, 4), textcoords="offset points", fontsize=8)
plt.xlabel("Total time (min)"); plt.ylabel("CO2 (kg)")
plt.title("Eco-routing trade-off (labels = CO2 weight)")
plt.grid(True); plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "eco_tradeoff.png", dpi=150)
plt.close()

R.plot_routes(all_routes[fast["setting"]], p, "Fastest routes", C.RESULTS_DIR / "eco_fastest_routes.png")
R.plot_routes(all_routes[eco["setting"]], p, "Lowest-CO2 routes", C.RESULTS_DIR / "eco_green_routes.png")
print("\nSaved: results/phase7_eco.csv, eco_routes.json, eco_tradeoff.png, "
      "eco_fastest_routes.png, eco_green_routes.png")
