"""Phase 4 - The scoring rule (cost function) and the simple baseline.

Before building a smart optimizer we need two things:
  1. A fair way to score any set of routes (evaluate() in routing_core.py).
  2. A simple 'beat this' reference, so we can later prove our optimizer is better.

We build two references:
  - Random: customers in random order (average of 500 tries).
  - Nearest neighbour: every bike always goes to the closest customer next.

Run from the project folder:
    py src/baseline.py
"""
import json
import numpy as np
import pandas as pd
import config as C
import routing_core as R

p = R.load_problem()
print(f"Customers: {p.n_customers} | Bikes: {p.n_vehicles} | "
      f"Capacity per bike: {p.capacity} | Total demand: {int(p.demand.sum())}")
print(f"Service time per customer: {R.SERVICE_MIN} min | "
      f"Imbalance weight: {R.W_BALANCE} | CO2 weight: {R.W_CO2}")

# Reference 1: random routes (average over 500 tries)
rng = np.random.default_rng(C.SEED)
tries = [R.evaluate(R.random_feasible(p, rng), p) for _ in range(500)]
rand = {k: float(np.mean([t[k] for t in tries]))
        for k in ["cost", "total_time_min", "total_km", "co2_kg",
                  "longest_route_min", "imbalance_min", "overload"]}

# Reference 2: nearest neighbour
nn_routes = R.nearest_neighbour(p)
nn = R.evaluate(nn_routes, p)

print("\n=== Nearest-neighbour routes ===")
for k, r in enumerate(nn_routes, 1):
    path = " -> ".join(["Depot"] + [p.names[i] for i in r] + ["Depot"])
    print(f"Bike {k}: {path}")
    print(f"        load {nn['loads'][k-1]}/{p.capacity} | "
          f"time {nn['route_times'][k-1]:.0f} min | distance {nn['route_kms'][k-1]:.1f} km")

cols = ["cost", "total_time_min", "total_km", "co2_kg",
        "longest_route_min", "imbalance_min", "overload"]
table = pd.DataFrame([{"method": "Random (average of 500)", **rand},
                      {"method": "Nearest neighbour", **{k: nn[k] for k in cols}}]).round(2)
print("\n=== Scores (lower is better) ===")
print(table.to_string(index=False))
print("\ntotal_time_min  = all travel plus service time of all bikes added up")
print("imbalance_min   = longest route minus shortest route (fairness)")
print("co2_kg          = uses an ASSUMED emission value, treat as an estimate")

gain = (rand["cost"] - nn["cost"]) / rand["cost"] * 100
print(f"\nNearest neighbour is {gain:.0f}% better than random routes on the cost score.")

C.RESULTS_DIR.mkdir(exist_ok=True)
table.to_csv(C.RESULTS_DIR / "phase4_baseline.csv", index=False)
pd.Series({"routes": nn_routes}).to_json(C.RESULTS_DIR / "baseline_routes.json")
R.plot_routes(nn_routes, p, "Nearest-neighbour baseline (Ajmer)",
              C.RESULTS_DIR / "baseline_routes.png")
print("Saved: results/phase4_baseline.csv, baseline_routes.json, baseline_routes.png")
print("\nPhase 4 done.")
