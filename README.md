# AI-Driven Delivery Route Optimization (Ajmer)

Final year project. Machine learning (XGBoost) predicts delivery travel times from
real Indian food-delivery data (Kaggle). A Genetic Algorithm with Simulated Annealing
then plans routes for several vehicles across real locations in Ajmer.

## Folder structure
```
route_opt/
  README.md              this file
  requirements.txt       Python libraries to install
  data/raw/              downloaded dataset (train.csv) and Ajmer locations
  data/processed/        cleaned data created by our code
  src/                   all Python code
  models/                trained models
  results/               charts and result tables
  docs/                  notes for each phase (reuse them in the report)
```

## Setup (one time)
1. Install Python 3.10 or newer (tick "Add python.exe to PATH" on Windows).
2. Install Visual Studio Code, then File > Open Folder > choose this folder.
3. Terminal > New Terminal, then run:  `pip install -r requirements.txt`
4. Put the dataset in `data/raw/train.csv` (see docs/01_data_source.md).
5. Check it: `python src/check_data.py`
