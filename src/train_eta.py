"""Phase 2 - Teach the computer to guess delivery time (the 'ETA model').

Steps:
  1. Load the clean data from Phase 1.
  2. Convert it to numbers (features.py).
  3. Split: 80% for learning, 20% kept hidden as an 'exam'.
  4. Train two simple guessers (baselines) and the XGBoost model.
  5. Compare them on the hidden exam, save charts and the model.

Run from the project folder:
    py src/train_eta.py
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")          # draw charts into files, no window needed
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config as C
from features import make_features

FEATURES_FILE = C.ROOT / "models" / "eta_features.json"

if not C.CLEAN_FILE.exists():
    print("Clean file not found. Run Phase 1 first:  py src/clean_data.py")
    raise SystemExit(1)

df = pd.read_csv(C.CLEAN_FILE)
X = make_features(df)
y = df["time_taken_min"]

# Safety: drop rows where a word was not recognised (should be none)
bad = X.isna().any(axis=1)
if bad.any():
    print(f"Note: removing {bad.sum()} rows with unknown traffic words.")
    X, y = X[~bad], y[~bad]

# 80% to learn from, 20% hidden for the exam
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=C.SEED)
print(f"Learning rows: {len(X_tr)} | Exam rows: {len(X_te)}")

# Guesser 1: ignores everything and always says the average time
pred_avg = np.full(len(y_te), y_tr.mean())

# Guesser 2: uses only distance (a straight line)
line = LinearRegression().fit(X_tr[["distance_km"]], y_tr)
pred_line = line.predict(X_te[["distance_km"]])

# Our model: XGBoost uses everything (traffic, weather, hour, vehicle...)
model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.9, random_state=C.SEED)
model.fit(X_tr, y_tr)
pred = model.predict(X_te)

rows = []
for name, p in [("Always the average", pred_avg),
                ("Distance only (line)", pred_line),
                ("XGBoost (our model)", pred)]:
    rows.append({"model": name,
                 "avg_error_min": round(mean_absolute_error(y_te, p), 2),
                 "r2": round(r2_score(y_te, p), 3)})
results = pd.DataFrame(rows)

print("\n=== Results on the hidden exam rows ===")
print("avg_error_min = how many minutes the guess is wrong, on average (lower is better)")
print("r2            = how much of the pattern was understood (1 = perfect, 0 = nothing)\n")
print(results.to_string(index=False))

train_err = mean_absolute_error(y_tr, model.predict(X_tr))
print(f"\nCheck for cheating: error on learning rows {train_err:.2f} min, "
      f"on exam rows {rows[2]['avg_error_min']} min")
print("(If the exam error is MUCH bigger, the model just memorised.)")

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nWhat the model pays attention to (top 8):")
print(imp.head(8).round(3).to_string())

# Charts and files
C.RESULTS_DIR.mkdir(exist_ok=True)
results.to_csv(C.RESULTS_DIR / "phase2_results.csv", index=False)

top = imp.head(10)[::-1]
plt.figure(figsize=(7, 4))
plt.barh(top.index, top.values)
plt.title("What matters most for delivery time")
plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "feature_importance.png", dpi=150)
plt.close()

plt.figure(figsize=(5, 5))
plt.scatter(y_te, pred, s=4, alpha=0.3)
lim = [0, max(y_te.max(), pred.max())]
plt.plot(lim, lim, "r--")
plt.xlabel("Real delivery time (min)")
plt.ylabel("Model's guess (min)")
plt.title("Guess vs real (closer to the red line = better)")
plt.tight_layout()
plt.savefig(C.RESULTS_DIR / "predicted_vs_real.png", dpi=150)
plt.close()

C.MODEL_FILE.parent.mkdir(exist_ok=True)
model.save_model(str(C.MODEL_FILE))
FEATURES_FILE.write_text(json.dumps(list(X.columns)))
print(f"\nSaved model  -> {C.MODEL_FILE}")
print(f"Saved charts -> {C.RESULTS_DIR}")
print("\nPhase 2 done.")
