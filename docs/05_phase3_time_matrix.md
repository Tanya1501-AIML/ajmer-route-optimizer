# Phase 3 - Travel-time map for Ajmer

## The idea (simple version)
The route finder will try thousands of route plans. For each plan it must add up the
travel times between stops. Asking the model every time would be slow, so we ask it once
for every pair of stops and write the answers in a table (a matrix). Row = from, column = to.

## What we did
1. Read data/raw/ajmer_locations.csv (1 depot + 20 customers, coordinates copied from
   Google Maps) and checked it: no empty rows, latitude/longitude not swapped, inside India.
2. Measured the straight-line distance between every pair of stops (haversine formula).
   This is the same kind of distance the model was trained on.
3. Chose a situation to plan for (hour, weekday, traffic, weather, vehicle) at the top of
   build_matrix.py. Default: Friday 6 PM, High traffic, sunny, motorcycle.
4. Ran a test: same conditions, only the distance changes (1, 2, 3 ... 15 km).
   A trustworthy time table must show longer trips taking longer.
5. Built the time table in one of two ways, chosen by the test:
   - MODEL: if the model clearly takes longer for longer distances, every trip time comes
     straight from the model.
   - HYBRID: if the model barely reacts to distance, we use the model only to learn how much
     traffic, weather and hour slow things down (compared with a calm trip), and combine that
     with distance / speed. The base speed (25 km/h) is an assumption.
6. Saved distance_matrix.csv, time_matrix.csv, ajmer_stops_clean.csv (data/processed)
   and a picture of the points (results/ajmer_points.png).

## Why the test matters
In the Kaggle data, distance alone explains little of the delivery time (r2 about 0.09), because
delivery time also contains waiting, preparation and handover. If we ignored this, every
stop would look equally far from every other stop and route optimization would be pointless.

## Notes for the report
- Demand and base speed are assumptions. Locations are real places in Ajmer.
- The situation (hour, traffic, weather) is a setting; the dashboard will let you change it.
- Straight-line distance is used. Road distance (OpenStreetMap) is a possible future improvement.
