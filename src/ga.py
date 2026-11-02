"""Phase 5 - Genetic Algorithm (the main optimizer).

Idea (copying evolution):
  1. Start with a population of random route sets.
  2. Score every one with R.evaluate() (lower cost = better).
  3. Pick good parents, mix them (crossover), tweak some (mutation).
  4. Keep the very best ones untouched (elitism). Repeat for many generations.

How a solution is stored (the 'chromosome'):
  customers are 1..N and the extra numbers N+1 .. N+K-1 are separators that cut
  the list into K bike routes.
  Example N=6, K=2:  [3, 1, 5, 7, 2, 6, 4]  ->  bike1=[3,1,5]  bike2=[2,6,4]
  Overloaded bikes are NOT forbidden; evaluate() adds a huge penalty, so the GA
  learns to avoid them.

Run from the project folder:
    py src/ga.py
"""
import json
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
POP_SIZE = 100
GENERATIONS = 300
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.25
TOURNAMENT_K = 3
ELITE = 2
SEED_WITH_NN = False   # True = put nearest-neighbour routes in the first population
# ----------------------------------------------------------------------

p = R.load_problem()
N = p.n_customers
K = p.n_vehicles
L = N + K - 1                      # chromosome length

random.seed(C.SEED)
np.random.seed(C.SEED)


def decode(chrom):
    routes, cur = [], []
    for g in chrom:
        if g > N:                  # separator -> next bike
            routes.append(cur)
            cur = []
        else:
            cur.append(int(g))
    routes.append(cur)
    return routes


def encode(routes):
    chrom = []
    for k, r in enumerate(routes):
        chrom.extend(r)
        if k < len(routes) - 1:
            chrom.append(N + 1 + k)
    return chrom


def cost_of(chrom):
    return R.evaluate(decode(chrom), p)["cost"]


def random_individual():
    genes = list(range(1, N + K))
    random.shuffle(genes)
    return genes


def mutate(chrom):
    c = chrom[:]
    i, j = sorted(random.sample(range(L), 2))
    r = random.random()
    if r < 0.4:                    # swap two genes
        c[i], c[j] = c[j], c[i]
    elif r < 0.8:                  # reverse a segment
        c[i:j + 1] = reversed(c[i:j + 1])
    else:                          # move one gene somewhere else
        g = c.pop(i)
        c.insert(j, g)
    return c


def tournament(pop, costs):
    idx = random.sample(range(len(pop)), TOURNAMENT_K)
    return pop[min(idx, key=lambda i: costs[i])]


def order_crossover(p1, p2):
    """Copy a slice of parent 1, fill the rest in parent 2's order (no duplicates)."""
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
    pop = [random_individual() for _ in range(POP_SIZE)]
    if SEED_WITH_NN:
        nn = encode(R.nearest_neighbour(p))
        pop[:5] = [nn] + [mutate(nn) for _ in range(4)]
    costs = [cost_of(c) for c in pop]
    history = []

    for gen in range(GENERATIONS):
        order = np.argsort(costs)
        pop = [pop[i] for i in order]
        costs = [costs[i] for i in order]
        history.append(float(costs[0]))

        new_pop = pop[:ELITE]
        while len(new_pop) < POP_SIZE:
            p1, p2 = tournament(pop, costs), tournament(pop, costs)
            child = order_crossover(p1, p2) if random.random() < CROSSOVER_RATE else p1[:]
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        costs = [cost_of(c) for c in pop]

        if gen % 25 == 0 or gen == GENERATIONS - 1:
            print(f"Generation {gen:3d} | best cost = {min(costs):.2f}")

    best = int(np.argmin(costs))
    history.append(float(costs[best]))
    return decode(pop[best]), history


start = time.time()
ga_routes, history = run_ga()
elapsed = time.time() - start

ga = R.evaluate(ga_routes, p)
nn_routes = R.nearest_neighbour(p)
nn = R.evaluate(nn_routes, p)

print("\n=== GA routes ===")
for k, r in enumerate(ga_routes, 1):
    path = " -> ".join(["Depot"] + [p.names[i] for i in r] + ["Depot"])
    print(f"Bike {k}: {path}")
    print(f"        load {ga['loads'][k-1]}/{p.capacity} | "
          f"time {ga['route_times'][k-1]:.0f} min | distance {ga['route_kms'][k-1]:.1f} km")

cols = ["cost", "total_time_min", "total_km", "co2_kg",
        "longest_route_min", "imbalance_min", "overload"]
table = pd.DataFrame([{"method": "Nearest neighbour", **{k: nn[k] for k in cols}},
                      {"method": "Genetic Algorithm", **{k: ga[k] for k in cols}}]).round(2)
print("\n=== Scores (lower is better) ===")
print(table.to_string(index=False))

gain = (nn["cost"] - ga["cost"]) / nn["cost"] * 100
print(f"\nGA is {gain:.1f}% better than nearest neighbour on the cost score.")
print(f"GA run time: {elapsed:.1f} seconds")
if ga["overload"] > 0:
    print("WARNING: GA solution still overloads a bike. Increase GENERATIONS or POP_SIZE.")

C.RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(C.RESULTS_DIR / "phase5_ga.csv", index=False)
with open(C.RESULTS_DIR / "ga_routes.json", "w") as f:
    json.dump({"routes": ga_routes, "cost": float(ga["cost"]),
               "history": history, "seconds": elapsed}, f)

plt.figure(figsize=(7, 4))
plt.plot(history, label="GA best cost")
plt.axhline(nn["cost"], color="red", linestyle="--", label="Nearest neighbour")
plt.xlabel("Generation"); plt.ylabel("Cost (lower is better)")
plt.title("GA: cost dropping over generations")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "ga_convergence.png", dpi=150)
plt.close()

R.plot_routes(ga_routes, p, "Genetic Algorithm routes (Ajmer)", C.RESULTS_DIR / "ga_routes.png")
print("Saved: results/phase5_ga.csv, ga_routes.json, ga_convergence.png, ga_routes.png")
print("\nPhase 5 done.")
