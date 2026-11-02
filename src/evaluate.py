"""Phase 10 - Evaluation: is our system really better?

We create many RANDOM delivery days (scenarios). In each one we change:
  - which customers have orders (about 60% to 100% of your Ajmer stops),
  - how big each order is (within the range of order sizes in your own data),
  - the weekday, the hour, the weather and the traffic (so travel times change).
For every scenario we build routes 3 ways and score them with the same cost function:
  1. Nearest neighbour  (the simple method),
  2. Genetic Algorithm  (GA),
  3. GA + Simulated Annealing (GA+SA).
Then we average over all scenarios and make tables and charts for the report.

Run from the project folder (after Phase 6):
    py src/evaluate.py          (takes a few minutes)
"""
import json
import math
import random
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import routing_core as R
from features import make_features, WEATHER_VALUES

# ----------------------------- settings -------------------------------
N_SCENARIOS = 30
N_BIKES = int(R.N_VEHICLES)
GA_POP, GA_GENS = 100, 300          # same as Phase 5
SA_STEPS = 40000                    # same as Phase 6
VEHICLE, MULTIPLE_DELIVERIES, FESTIVAL = "motorcycle", 1, "No"
BASE_SPEED_KMPH = 25
# ----------------------------------------------------------------------
FEATURES_FILE = C.ROOT / "models" / "eta_features.json"
METRICS = ["cost", "total_time_min", "total_km", "co2_kg", "imbalance_min"]
LABELS = {"nn": "Nearest neighbour", "ga": "Genetic Algorithm", "gasa": "GA + SA"}

model = xgb.XGBRegressor()
model.load_model(str(C.MODEL_FILE))
FN = json.loads(FEATURES_FILE.read_text())
stops = pd.read_csv(R.STOPS_FILE)
D0 = pd.read_csv(R.DIST_FILE, index_col=0).to_numpy(dtype=float)
N_ALL = len(stops) - 1                      # number of customers in YOUR file
MAX_CUSTOMERS = N_ALL
MIN_CUSTOMERS = max(3, int(round(0.6 * N_ALL)))
DEMAND_MIN = int(stops["demand"].iloc[1:].min())
DEMAND_MAX = int(stops["demand"].iloc[1:].max())
if N_ALL < 4:
    raise SystemExit(f"Only {N_ALL} customers found in the stops file. Add more customers first.")
print(f"Stops file: {N_ALL} customers, order sizes {DEMAND_MIN} to {DEMAND_MAX}. "
      f"Each random day uses {MIN_CUSTOMERS} to {MAX_CUSTOMERS} of them.")


# ---------------------------------------------------------------- times
def predict_minutes(dists, hour, weekday, weather, traffic):
    d = pd.DataFrame({"distance_km": np.asarray(dists, dtype=float)})
    d["hour"], d["weekday"], d["is_weekend"] = hour, weekday, int(weekday >= 5)
    d["traffic"], d["weather"], d["vehicle"] = traffic, weather, VEHICLE
    d["multiple_deliveries"], d["festival"] = MULTIPLE_DELIVERIES, FESTIVAL
    return model.predict(make_features(d)[FN])


def build_time_matrix(D, hour, weekday, weather, traffic):
    """Same idea as Phase 3 (model, or hybrid if the distance effect is weak)."""
    extra = float(predict_minutes([12], hour, weekday, weather, traffic)[0] -
                  predict_minutes([2], hour, weekday, weather, traffic)[0])
    speed = 10 / (extra / 60) if extra > 0 else float("inf")
    if speed > 45:
        ref = predict_minutes([8], 11, weekday, "Sunny", "Low")[0]
        now = predict_minutes([8], hour, weekday, weather, traffic)[0]
        T = D / BASE_SPEED_KMPH * 60 * (now / ref)
    else:
        T = predict_minutes(D.ravel(), hour, weekday, weather, traffic).reshape(D.shape)
    T = np.asarray(T, dtype=float)
    np.fill_diagonal(T, 0)
    return T


# ------------------------------------------------------------ optimizer
def cost_of(routes, p):
    return R.evaluate(routes, p)["cost"]


def neighbour(routes):
    new = [list(r) for r in routes]
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


def run_sa(start, p, steps, t_start=20.0, t_end=0.01):
    cur, cur_c = start, cost_of(start, p)
    best, best_c = [list(r) for r in cur], cur_c
    alpha = (t_end / t_start) ** (1.0 / steps)
    T = t_start
    for _ in range(steps):
        cand = neighbour(cur)
        cc = cost_of(cand, p)
        d = cc - cur_c
        if d < 0 or random.random() < math.exp(-d / T):
            cur, cur_c = cand, cc
            if cur_c < best_c:
                best, best_c = [list(r) for r in cur], cur_c
        T *= alpha
    return best


def run_ga(p, pop_size, gens):
    n, k = p.n_customers, p.n_vehicles
    L = n + k - 1

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

    def ox(p1, p2):
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
    costs = [cost_of(decode(c), p) for c in pop]
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
        costs = [cost_of(decode(c), p) for c in pop]
    return decode(pop[int(np.argmin(costs))])


# ------------------------------------------------------------- scenarios
def make_scenario(i):
    rng = np.random.default_rng(C.SEED + i)
    n_c = int(rng.integers(MIN_CUSTOMERS, MAX_CUSTOMERS + 1))
    idx = sorted(rng.choice(np.arange(1, N_ALL + 1), n_c, replace=False).tolist())
    nodes = [0] + idx
    D = D0[np.ix_(nodes, nodes)]
    cond = {"weekday": int(rng.integers(0, 7)), "hour": int(rng.integers(0, 24)),
            "weather": str(rng.choice(WEATHER_VALUES)),
            "traffic": str(rng.choice(["Low", "Medium", "High", "Jam"]))}
    T = build_time_matrix(D, cond["hour"], cond["weekday"], cond["weather"], cond["traffic"])
    demand = np.r_[0, rng.integers(DEMAND_MIN, DEMAND_MAX + 1, n_c)].astype(int)
    cap = max(int(math.ceil(1.3 * demand.sum() / N_BIKES)), int(demand.max()))
    p = R.Problem(list(stops["name"].iloc[nodes]), stops["lat"].to_numpy()[nodes],
                  stops["lon"].to_numpy()[nodes], demand, T, D, N_BIKES, cap)
    return p, cond, n_c


# ----------------------------------------------------------------- main
rows = []
t_all = time.time()
for i in range(N_SCENARIOS):
    p, cond, n_c = make_scenario(i)
    random.seed(C.SEED + i)
    np.random.seed(C.SEED + i)

    t0 = time.perf_counter()
    nn_routes = R.nearest_neighbour(p)
    s_nn = time.perf_counter() - t0

    t0 = time.perf_counter()
    ga_routes = run_ga(p, GA_POP, GA_GENS)
    s_ga = time.perf_counter() - t0

    t0 = time.perf_counter()
    sa_routes = run_sa(ga_routes, p, SA_STEPS)
    s_sa = s_ga + (time.perf_counter() - t0)

    for key, routes, sec in [("nn", nn_routes, s_nn), ("ga", ga_routes, s_ga), ("gasa", sa_routes, s_sa)]:
        m = R.evaluate(routes, p)
        rows.append({"scenario": i + 1, "method": key, "customers": n_c, **cond,
                     **{k: m[k] for k in METRICS}, "overload": m["overload"],
                     "longest_route_min": m["longest_route_min"], "seconds": sec})
    r = rows[-3:]
    print(f"Scenario {i + 1:2d}/{N_SCENARIOS} | {n_c} customers, {cond['hour']:2d}:00, "
          f"{cond['weather']:10s}, {cond['traffic']:6s} | cost NN {r[0]['cost']:7.1f} "
          f"-> GA {r[1]['cost']:7.1f} -> GA+SA {r[2]['cost']:7.1f}")

df = pd.DataFrame(rows)
C.RESULTS_DIR.mkdir(exist_ok=True)
df.round(3).to_csv(C.RESULTS_DIR / "phase10_scenarios.csv", index=False)

# ---------------------------------------------------------------- tables
wide = {k: df[df["method"] == k].set_index("scenario") for k in LABELS}
summary = []
for k, label in LABELS.items():
    d = wide[k]
    row = {"method": label}
    for m in METRICS + ["longest_route_min", "seconds"]:
        row[m] = f"{d[m].mean():.2f} +/- {d[m].std():.2f}"
    row["infeasible_scenarios"] = int((d["overload"] > 0).sum())
    summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(C.RESULTS_DIR / "phase10_summary.csv", index=False)
print("\n=== Average over all scenarios (mean +/- std, lower is better) ===")
print(summary.to_string(index=False))

# fair comparison: only scenarios where the simple method is also feasible
ok = (wide["nn"]["overload"] == 0) & (wide["gasa"]["overload"] == 0) & (wide["ga"]["overload"] == 0)
n_ok = int(ok.sum())
rng = np.random.default_rng(C.SEED)


def pct(a, b):                      # % improvement of b over a (positive = b is better)
    return ((a - b) / a * 100)[ok].to_numpy()


def boot(x, n=3000):
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


imp_rows, imp_data = [], {}
for m, nice in [("cost", "Cost score"), ("total_time_min", "Total time"),
                ("total_km", "Distance"), ("co2_kg", "CO2")]:
    for key in ["ga", "gasa"]:
        x = pct(wide["nn"][m], wide[key][m])
        mean, lo, hi = boot(x)
        imp_data[(m, key)] = x
        imp_rows.append({"metric": nice, "method": LABELS[key],
                         "better than NN by (%)": round(mean, 2),
                         "95% CI": f"[{lo:.2f}, {hi:.2f}]",
                         "wins": f"{int((x > 1e-9).sum())}/{n_ok}"})
imp = pd.DataFrame(imp_rows)
imp.to_csv(C.RESULTS_DIR / "phase10_improvement_vs_nn.csv", index=False)
print(f"\n=== Improvement over nearest neighbour (positive = better), {n_ok} feasible scenarios ===")
print(imp.to_string(index=False))

sa_gain = pct(wide["ga"]["cost"], wide["gasa"]["cost"])
sa_mean, sa_lo, sa_hi = boot(sa_gain)
imb_diff = (wide["nn"]["imbalance_min"] - wide["gasa"]["imbalance_min"])[ok].to_numpy()
imb_mean, imb_lo, imb_hi = boot(imb_diff)
cost_x = imp_data[("cost", "gasa")]
c_mean, c_lo, c_hi = boot(cost_x)

# ---------------------------------------------------------------- verdict
lines = [
    f"Tested on {N_SCENARIOS} random delivery days ({n_ok} where all methods carried every load).",
    f"GA + SA beat nearest neighbour on the cost score in {int((cost_x > 1e-9).sum())} of {n_ok} days.",
    f"Average cost improvement: {c_mean:.1f}% (95% CI {c_lo:.1f}% to {c_hi:.1f}%).",
    f"Total travel time: {imp_data[('total_time_min', 'gasa')].mean():.1f}% lower, "
    f"distance: {imp_data[('total_km', 'gasa')].mean():.1f}% lower, "
    f"CO2: {imp_data[('co2_kg', 'gasa')].mean():.1f}% lower (all on average).",
    f"Fairness: the gap between the longest and shortest route is {abs(imb_mean):.1f} min "
    f"{'smaller' if imb_mean >= 0 else 'LARGER'} than nearest neighbour on average "
    f"(95% CI of the difference {imb_lo:.1f} to {imb_hi:.1f} min; positive = fairer).",
    f"SA on top of GA added {sa_mean:.2f}% (95% CI {sa_lo:.2f}% to {sa_hi:.2f}%).",
    f"Run time per day: nearest neighbour {wide['nn']['seconds'].mean() * 1000:.1f} ms, "
    f"GA {wide['ga']['seconds'].mean():.1f} s, GA+SA {wide['gasa']['seconds'].mean():.1f} s.",
]
if c_lo > 0 and (cost_x > 1e-9).mean() >= 0.8:
    lines.append("ANSWER: Yes. Our system is clearly better than the simple method "
                 "(the confidence interval is above zero and it wins on most days).")
elif c_mean > 0:
    lines.append("ANSWER: Mostly yes, but the evidence is not very strong on this sample.")
else:
    lines.append("ANSWER: No clear advantage on this sample. Check settings before the report.")
if imb_hi < 0:
    lines.append("NOTE: our routes are LESS evenly balanced between bikes. The cost score trades "
                 "some fairness for less time and distance. Say this in the report, or raise "
                 "W_BALANCE in routing_core.py if fairness matters more.")
verdict = "\n".join(lines)
print("\n=== So, is the system better? ===")
print(verdict)
(C.RESULTS_DIR / "phase10_verdict.txt").write_text(verdict, encoding="utf-8")

# ---------------------------------------------------------------- charts
fig, ax = plt.subplots(figsize=(6.5, 4.2))
means = [wide[k]["cost"].mean() for k in LABELS]
stds = [wide[k]["cost"].std() for k in LABELS]
ax.bar(list(LABELS.values()), means, yerr=stds, capsize=6, color=["tab:gray", "tab:blue", "tab:green"])
ax.set_ylabel("Cost score (lower is better)")
ax.set_title(f"Average cost over {N_SCENARIOS} random days")
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "phase10_cost_bars.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
box = [imp_data[(m, "gasa")] for m in ["cost", "total_time_min", "total_km", "co2_kg"]]
axes[0].boxplot(box, tick_labels=["Cost", "Time", "Distance", "CO2"])
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_ylabel("% better than nearest neighbour")
axes[0].set_title("GA + SA vs nearest neighbour (per day)")
axes[0].grid(True, axis="y")
axes[1].boxplot([wide[k]["imbalance_min"] for k in LABELS], tick_labels=list(LABELS.values()))
axes[1].set_ylabel("Longest minus shortest route (min)")
axes[1].set_title("Fairness between bikes (lower is better)")
axes[1].grid(True, axis="y")
plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "phase10_improvement_boxplots.png", dpi=150)
plt.close()

order = wide["nn"]["cost"].sort_values().index
plt.figure(figsize=(8, 4.2))
for k, col in zip(LABELS, ["tab:gray", "tab:blue", "tab:green"]):
    plt.plot(range(1, len(order) + 1), wide[k]["cost"].loc[order].to_numpy(), "-o", label=LABELS[k],
             color=col, markersize=4)
plt.xlabel("Day (sorted from easiest to hardest for nearest neighbour)")
plt.ylabel("Cost score")
plt.title("Cost on every random day")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "phase10_cost_per_day.png", dpi=150)
plt.close()

print(f"\nTotal time: {(time.time() - t_all) / 60:.1f} min")
print("Saved in results/: phase10_scenarios.csv, phase10_summary.csv, phase10_improvement_vs_nn.csv, "
      "phase10_verdict.txt, phase10_cost_bars.png, phase10_improvement_boxplots.png, phase10_cost_per_day.png")
print("\nPhase 10 done.")
