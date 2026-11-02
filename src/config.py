"""Central settings. Every other file imports paths and constants from here.
If you want to change a file name or a setting, change it only here."""
from pathlib import Path

# Project root = the folder that contains README.md (one level above src/)
ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "train.csv"             # Kaggle food delivery data
AJMER_FILE = ROOT / "data" / "raw" / "ajmer_locations.csv"  # our own Ajmer points (later)
CLEAN_FILE = ROOT / "data" / "processed" / "deliveries_clean.csv"
MODEL_FILE = ROOT / "models" / "eta_xgb.json"
RESULTS_DIR = ROOT / "results"

SEED = 42  # fixed random seed so results are repeatable
