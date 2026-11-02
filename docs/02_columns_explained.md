# What the columns mean (simple version)

Each row = one delivery that already happened.

| Column | Meaning |
|--------|---------|
| ID | order number |
| Delivery_person_ID / Age / Ratings | the rider's ID, age and rating |
| Restaurant_latitude / longitude | where the order started |
| Delivery_location_latitude / longitude | where it was delivered |
| Order_Date | date of the order |
| Time_Orderd | time the order was placed (spelling is as in the dataset) |
| Time_Order_picked | time the rider picked it up |
| Weatherconditions | weather during delivery |
| Road_traffic_density | Low, Medium, High or Jam |
| Vehicle_condition | condition of the vehicle (a number) |
| Type_of_order | Snack, Meal, Drinks, Buffet |
| Type_of_vehicle | motorcycle, scooter, electric scooter |
| multiple_deliveries | how many other deliveries the rider was carrying |
| Festival | whether it was a festival day |
| City | Urban, Metropolitian or Semi-Urban (spelling as in the dataset) |
| Time_taken(min) | minutes the delivery took. THIS is what we teach the model to guess |

Keep the odd spellings (Time_Orderd, Metropolitian) exactly as they are in the code.
