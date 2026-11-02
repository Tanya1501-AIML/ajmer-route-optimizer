# Data source

## Dataset used
**Food Delivery Dataset** (Kaggle, uploaded by Gaurav Malik)

- Page: https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset
- Originally released by HackerEarth for a machine learning competition.
- About 45,000 deliveries across 21 cities in India.
- Each row is one past delivery: restaurant and delivery coordinates, order time,
  weather, road traffic density, vehicle type, and the delivery time in minutes.

## How to download (one time)
1. Log in to kaggle.com (free account).
2. Open the page linked above and click Download. You get a zip file.
3. Extract it. Find the file that has a `Time_taken(min)` column.
4. Copy that file into this project at `data/raw/` and rename it `train.csv`.
5. Check it: `python src/check_data.py`

## Our own data: Ajmer locations (added later)
One depot and about 20 customer locations in Ajmer, with coordinates copied from
Google Maps, saved as `data/raw/ajmer_locations.csv`.

## What is real and what is assumed
| Item | Real or assumed |
|------|-----------------|
| Delivery times, traffic, weather, distances used to train the model | Real (Kaggle dataset) |
| Ajmer depot and customer locations | Real places, chosen by us from Google Maps |
| Number of orders, customer demand, vehicle capacity | Assumed |
| CO2 emission per km | Assumed typical value, to be cited from a published source |

Write these assumptions clearly in the report.
