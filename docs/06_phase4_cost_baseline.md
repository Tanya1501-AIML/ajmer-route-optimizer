# Phase 4 - Scoring rule (cost function) and baseline

## The idea (simple version)
To decide which routes are "good", the computer needs a score for every set of routes,
like marks in an exam. Lower cost = better routes. We also need a simple reference
method to beat. If our smart optimizer cannot beat it, the smart optimizer is not worth much.

## The setup
- 1 depot, the Ajmer customers, 3 bikes (N_VEHICLES in routing_core.py).
- Every bike starts at the depot, visits some customers, and returns to the depot.
- Each customer has a demand (packets). A bike has a capacity (max packets).
  Capacity is chosen automatically with 30% spare room.

## The cost function (evaluate in routing_core.py)
cost = total time of all bikes
     + W_BALANCE x (longest route time - shortest route time)
     + W_CO2 x estimated CO2
     + a huge penalty for every packet above a bike's capacity

- Total time = travel time from the Phase 3 table + 5 minutes service time at each customer.
- Balance term: stops one rider from getting a very long route while another gets a short one.
- CO2 term is switched off for now (W_CO2 = 0). We switch it on in Phase 7 (eco-routing).
- The overload penalty makes breaking the capacity rule so expensive that it never wins.

## The baselines (baseline.py)
- Random: customers in random order, filling one bike after another (average of 500 tries).
- Nearest neighbour: each bike keeps driving to the closest customer it can still carry.

## What to look for in the result
- Nearest neighbour usually beats random on total time.
- But it is often unfair: the first bikes fill up and the last bike gets very little work.
  This shows up as a big imbalance_min. Our Genetic Algorithm should fix this.

## Assumptions to write in the report
- Service time per customer: 5 minutes (assumed).
- Road distance = straight-line distance x 1.3 (used only for CO2).
- CO2 per km is an assumed placeholder; a cited value will be added in Phase 7.
- Customer demand and bike capacity are assumed.
