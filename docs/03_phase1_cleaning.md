# Phase 1 - Cleaning the data

## Why we clean
The raw file is real data, so it is messy. A model learns wrong things from messy data,
the same way a student learns wrong things from a notebook full of mistakes.

## What was messy (found by check_data.py)
| Problem | Example | What we did |
|---------|---------|-------------|
| Extra spaces in text | "High ", "motorcycle " | removed the spaces |
| Missing values written as text | "NaN " in several columns | turned into real empty values |
| Words stuck to numbers | "(min) 24" | kept only the number 24 |
| Words stuck to labels | "conditions Sunny" | kept only "Sunny" |
| Invalid locations | latitude/longitude 0 | removed those rows |

## New columns we created
- distance_km: straight-line distance between restaurant and delivery point (haversine formula)
- hour: hour of the day the order was placed (0-23)
- weekday: day of the week (0 = Monday ... 6 = Sunday)
- is_weekend: 1 if Saturday or Sunday, else 0

## Columns we kept (known before a delivery starts)
distance_km, hour, weekday, is_weekend, traffic, weather, vehicle, multiple_deliveries,
festival, and the answer column time_taken_min.

We left out the rider's age and rating, and the pickup time. We will not know these
for a new Ajmer order, and the model must only use information available in advance.

## Rows removed
The script prints how many rows are left after each step. Write those numbers in the report.

## Notes for the report
- Distance is a straight-line distance computed from coordinates, not a GPS-tracked road distance.
- Rows with missing values were removed (not guessed), to keep the data honest.
