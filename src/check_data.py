"""Step 0 - Look at the downloaded dataset BEFORE changing anything.

Checks that the file is in the right place, has the right columns,
and shows what is messy (so we know what to clean in Phase 1).

Run from the project folder:
    python src/check_data.py
"""
import sys
import pandas as pd
import config as C

EXPECTED = [
    "ID", "Delivery_person_ID", "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Order_Date", "Time_Orderd", "Time_Order_picked", "Weatherconditions",
    "Road_traffic_density", "Vehicle_condition", "Type_of_order",
    "Type_of_vehicle", "multiple_deliveries", "Festival", "City",
    "Time_taken(min)",
]

if not C.RAW_FILE.exists():
    print("File not found:", C.RAW_FILE)
    print("Put the dataset file in the folder data/raw/ and name it train.csv")
    print("See docs/01_data_source.md for the steps.")
    sys.exit(1)

df = pd.read_csv(C.RAW_FILE)
print("Rows   :", len(df))
print("Columns:", len(df.columns))

missing = [c for c in EXPECTED if c not in df.columns]
if missing:
    print("\nProblem: these expected columns are missing:", missing)
    print("Columns found in your file:", list(df.columns))
    sys.exit(1)

print("\nFirst 5 rows (main columns):")
show = ["Restaurant_latitude", "Delivery_location_latitude", "Time_Orderd",
        "Weatherconditions", "Road_traffic_density", "Type_of_vehicle",
        "Time_taken(min)"]
print(df[show].head())

print("\n--- What is messy? (we fix this in Phase 1) ---")

print("\n1. Empty values per column (only columns that have some):")
empty = df.isna().sum()
print(empty[empty > 0])

print("\n2. Hidden 'NaN' written as text (with spaces), per column:")
for col in df.columns:
    n = (df[col].astype(str).str.strip() == "NaN").sum()
    if n > 0:
        print(f"   {col}: {n}")

print("\n3. Target column looks like this (words stuck in front of the number):")
print("  ", list(df["Time_taken(min)"].head(3)))

print("\n4. Weather values:", list(df["Weatherconditions"].dropna().unique()[:8]))
print("5. Traffic values:", list(df["Road_traffic_density"].dropna().unique()))
print("6. Vehicle values:", list(df["Type_of_vehicle"].dropna().unique()))

bad = ((df["Restaurant_latitude"].abs() < 1) |
       (df["Delivery_location_latitude"].abs() < 1)).sum()
print(f"\n7. Rows with a (near) zero latitude, i.e. invalid location: {bad}")

print("\nDone. If this ran without a red error, you are ready for Phase 1.")
