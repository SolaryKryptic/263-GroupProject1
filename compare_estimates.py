"""
compare_estimates.py

Ranks demand-estimation methods by delivered / required % on the held-out
test weeks (15-28 June 2026, the last 2 of the 8 data weeks).
"""

import glob
import os

import numpy as np
import pandas as pd

DEMAND_FILE = "FoodstuffsDemand2026.csv"
ESTIMATES_DIR = "estimates"
TEST_START = pd.Timestamp("2026-06-14")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Weighting for the Accuracy% metric: running short is judged worse than
# having spare capacity, so understock error counts at full weight and
# overstock error counts at half weight.
UNDERSTOCK_WEIGHT = 1.0
OVERSTOCK_WEIGHT = 0.5

# --- Cost model (from the project brief) ---
# Understock: a shortfall pallet is assumed to require emergency capacity
# wet-leased from Linfox at $1400 per 2-hour block. Assuming a similar
# ~16-pallet capacity per block gives an approximate per-pallet cost.
TRUCK_CAPACITY = 16
WETLEASE_COST_PER_BLOCK = 1400
UNDERSTOCK_COST_PER_PALLET = WETLEASE_COST_PER_BLOCK / TRUCK_CAPACITY  # ~$87.50/pallet

# Overstock: spare pallets on a truck already running the route cost
# ~nothing, UNLESS they push the route over the 16-pallet truck capacity
# and force an extra truck/trip - which depends on the (not yet solved)
# routing plan. Left at 0 as a placeholder; revisit once routes are known.
OVERSTOCK_COST_PER_PALLET = 0.0


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

	test = long_data[long_data["Date"] > TEST_START]
	return test[test["DayOfWeek"] != "Sunday"].copy()


def load_estimate(path):
	df = pd.read_csv(path)
	est = df[["Supermarket"] + WEEKDAYS[:-1]].set_index("Supermarket")

	for day in WEEKDAYS[:-1]:
		if day not in est.columns:
			print(f"  WARNING: {os.path.basename(path)} has no '{day}' column -> treated as 0")
			est[day] = 0
	return est


def compare_method(test, est):
	per_day = []
	delivered_total = 0
	required_total = 0
	weighted_error_total = 0
	estimated_total = 0
	total_cost = 0

	for day in WEEKDAYS[:-1]:
		sub = test[test["DayOfWeek"] == day].copy()
		sub = sub.merge(est[[day]].reset_index(), on="Supermarket", how="left")

		missing = sub[sub[day].isna()]
		if len(missing):
			stores = ", ".join(sorted(missing["Supermarket"].unique())[:5])
			print(f"  WARNING: {day}: {len(missing)} store-days missing from estimate ({stores}...)")
		sub[day] = sub[day].fillna(0)

		sub["Delivered"] = np.minimum(sub["Demand"], sub[day])
		delivered_total += sub["Delivered"].sum()
		required_total += sub["Demand"].sum()
		estimated_total += sub[day].sum()

		# error > 0 -> overstocked that store-day; error < 0 -> understocked
		error = sub[day] - sub["Demand"]
		understock = (-error).clip(lower=0)
		overstock = error.clip(lower=0)
		weighted_error_total += (UNDERSTOCK_WEIGHT * understock + OVERSTOCK_WEIGHT * overstock).sum()
		total_cost += (understock * UNDERSTOCK_COST_PER_PALLET + overstock * OVERSTOCK_COST_PER_PALLET).sum()

		day_pct = 100 * sub["Delivered"].sum() / sub["Demand"].sum() if sub["Demand"].sum() else np.nan
		per_day.append((sub["Date"].iloc[0], day_pct))

	overall_pct = 100 * delivered_total / required_total if required_total else np.nan
	# Accuracy%: penalizes understock at full weight and overstock at half
	# weight, so it can't be gamed by just over-ordering (like Delivered% can).
	accuracy_pct = 100 * (1 - weighted_error_total / required_total) if required_total else np.nan
	excess_pct = 100 * (estimated_total - delivered_total) / required_total if required_total else np.nan

	per_day_df = pd.DataFrame(per_day, columns=["Date", "Pct"])
	worst = per_day_df.sort_values("Pct").iloc[0]
	return (overall_pct, accuracy_pct, excess_pct, worst["Pct"], worst["Date"].strftime("%d/%m"),
	        required_total, total_cost)


def main():
	os.makedirs(ESTIMATES_DIR, exist_ok=True)

	test = load_test_demand()

	files = sorted(glob.glob(os.path.join(ESTIMATES_DIR, "estimate_*.csv")))
	if not files:
		print(f"No estimate_*.csv files found in '{ESTIMATES_DIR}'.")
		return

	rows = []
	for path in files:
		name = os.path.basename(path).removeprefix("estimate_").removesuffix(".csv")
		print(f"Comparing {name} ...")
		est = load_estimate(path)
		overall, accuracy, excess, worst_pct, worst_date, required, cost = compare_method(test, est)
		rows.append(
			{
				"Method": name,
				"Est. Cost ($)": int(round(cost)),
				"Delivered %": round(overall, 1),
				"Accuracy %": round(accuracy, 1),
				"Excess %": round(excess, 1),
				"Worst day %": round(worst_pct, 1),
				"Worst day": worst_date,
				"Test pallets (required)": int(required),
			}
		)

	result = pd.DataFrame(rows).sort_values("Est. Cost ($)", ascending=True).reset_index(drop=True)
	result.index += 1

	print(f"\nCost model: understock ${UNDERSTOCK_COST_PER_PALLET:.2f}/pallet "
	      f"(Linfox wet-lease, ${WETLEASE_COST_PER_BLOCK}/2hr block / {TRUCK_CAPACITY} pallets), "
	      f"overstock ${OVERSTOCK_COST_PER_PALLET:.2f}/pallet (placeholder - routing not yet solved)\n")
	print("Ranking by estimated cost over the test weeks (lower is better):\n")
	print(result.to_string(index=True))


if __name__ == "__main__":
	main()