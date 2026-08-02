"""
compare_estimates.py

Ranks demand-estimation methods by delivered / required % on the held-out
test weeks (15-28 June 2026, the last 2 of the 8 data weeks).

===================================== INPUT =====================================
Drop one CSV per method into the estimates/ folder:

    estimates/estimate_<name>.csv

Required columns:
    Supermarket   store name, must match FoodstuffsDemand2026.csv
    Monday        estimated pallets for every Monday
    Tuesday       estimated pallets for every Tuesday
    Wednesday     estimated pallets for every Wednesday
    Thursday      estimated pallets for every Thursday
    Friday        estimated pallets for every Friday
    Saturday      estimated pallets for every Saturday

A Sunday column is optional and ignored (stores are closed Sundays).
Missing stores/weekdays are treated as 0 with a warning.

===================================== METRIC ====================================
For each method:

    delivered% = 100 * sum(min(estimate, actual)) / sum(actual)

summed over every store-day in the test weeks (Sundays excluded).

A method that always delivers exactly the required amount scores 100%.
Under-estimating (actual > estimate) reduces the score; over-estimating
does not help because delivery is capped at the actual demand. This is
purely a measure of unmet demand (100 - delivered%).

===================================== RUN ======================================
    python compare_estimates.py
"""

import glob
import os

import numpy as np
import pandas as pd

DEMAND_FILE = "FoodstuffsDemand2026.csv"
ESTIMATES_DIR = "estimates"
TEST_START = pd.Timestamp("2026-06-14")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_test_demand():
	demand = pd.read_csv(DEMAND_FILE)
	long_data = demand.melt(
		id_vars="Supermarket",
		value_vars=demand.columns[1:],
		var_name="Date",
		value_name="Demand",
	)
	long_data["Date"] = pd.to_datetime(long_data["Date"], dayfirst=True)
	long_data["DayOfWeek"] = long_data["Date"].dt.day_name()

	# Test weeks are everything after the 6 training weeks, Sundays excluded.
	test = long_data[long_data["Date"] > TEST_START]
	return test[test["DayOfWeek"] != "Sunday"].copy()


def load_estimate(path):
	df = pd.read_csv(path)
	est = df[["Supermarket"] + WEEKDAYS[:-1]].set_index("Supermarket")

	# Any weekday column missing from the file is treated as 0 for that store.
	for day in WEEKDAYS[:-1]:
		if day not in est.columns:
			print(f"  WARNING: {os.path.basename(path)} has no '{day}' column -> treated as 0")
			est[day] = 0
	return est


def compare_method(test, est):
	# Join each test store-day to the matching weekday estimate.
	per_day = []
	delivered_total = 0
	required_total = 0

	for day in WEEKDAYS[:-1]:
		sub = test[test["DayOfWeek"] == day].copy()
		sub = sub.merge(est[[day]].reset_index(), on="Supermarket", how="left")

		# Stores missing from the estimate file count as 0 delivered (with warning).
		missing = sub[sub[day].isna()]
		if len(missing):
			stores = ", ".join(sorted(missing["Supermarket"].unique())[:5])
			print(f"  WARNING: {day}: {len(missing)} store-days missing from estimate ({stores}...)")
		sub[day] = sub[day].fillna(0)

		sub["Delivered"] = np.minimum(sub["Demand"], sub[day])
		delivered_total += sub["Delivered"].sum()
		required_total += sub["Demand"].sum()

		day_pct = 100 * sub["Delivered"].sum() / sub["Demand"].sum() if sub["Demand"].sum() else np.nan
		per_day.append((sub["Date"].iloc[0], day_pct))

	overall_pct = 100 * delivered_total / required_total if required_total else np.nan

	per_day_df = pd.DataFrame(per_day, columns=["Date", "Pct"])
	worst = per_day_df.sort_values("Pct").iloc[0]
	return overall_pct, worst["Pct"], worst["Date"].strftime("%d/%m"), required_total


def main():
	os.makedirs(ESTIMATES_DIR, exist_ok=True)

	test = load_test_demand()

	files = sorted(glob.glob(os.path.join(ESTIMATES_DIR, "estimate_*.csv")))
	if not files:
		print(f"No estimate_*.csv files found in '{ESTIMATES_DIR}'.")
		print("Add one file per method, e.g. estimates/estimate_<name>.csv")
		print("(see the file header for the required columns).")
		return

	rows = []
	for path in files:
		name = os.path.basename(path).removeprefix("estimate_").removesuffix(".csv")
		print(f"Comparing {name} ...")
		est = load_estimate(path)
		overall, worst_pct, worst_date, required = compare_method(test, est)
		rows.append(
			{
				"Method": name,
				"Delivered %": round(overall, 1),
				"Worst day %": round(worst_pct, 1),
				"Worst day": worst_date,
				"Test pallets (required)": int(required),
			}
		)

	result = pd.DataFrame(rows).sort_values("Delivered %", ascending=False).reset_index(drop=True)
	result.index += 1

	print("\nRanking by delivered/required % (higher is better):\n")
	print(result.to_string(index=True))


if __name__ == "__main__":
	main()
