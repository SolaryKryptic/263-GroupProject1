"""
compare_estimates.py

Ranks demand-estimation methods on the held-out test weeks (15-28 June 2026,
the last 2 of the 8 data weeks).

Each method is scored on the DOLLAR cost of its estimation errors, using the
cost parameters from the problem statement:
  - over-delivery (estimate > actual): extra unload time at $220/hr
  - under-delivery (estimate < actual): wet-lease top-up at $1400 / 2h block
    (a block unloads ~6.7 pallets at 18 min each), or shedding the store if
    that is cheaper ($1500 Pak 'n Save / $800 other)
  - feasibility: max % of stores shed on any day vs the 20% cap

Also reports delivered/required %, excess % and the worst test day.
"""

import glob
import os

import numpy as np
import pandas as pd

DEMAND_FILE = "FoodstuffsDemand2026.csv"
LOCATIONS_FILE = "FoodstuffsLocations.csv"
ESTIMATES_DIR = "estimations"
TEST_START = pd.Timestamp("2026-06-14")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Cost parameters from the problem statement.
UNLOAD_MIN_PER_PALLET = 18
TRUCK_COST_PER_HOUR = 220.0
OVER_COST_PER_PALLET = UNLOAD_MIN_PER_PALLET / 60 * TRUCK_COST_PER_HOUR  # $66 per over-delivered pallet

WET_LEASE_BLOCK_COST = 1400.0
WET_LEASE_BLOCK_MIN = 120  # 2 hours on-duty
WET_LEASE_PALLETS_PER_BLOCK = WET_LEASE_BLOCK_MIN / UNLOAD_MIN_PER_PALLET  # ~6.7 pallets
UNDER_COST_PER_PALLET = WET_LEASE_BLOCK_COST / WET_LEASE_PALLETS_PER_BLOCK  # $210 per short pallet

SHED_COST_PAKNSAVE = 1500.0
SHED_COST_OTHER = 800.0


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
	test = test[test["DayOfWeek"] != "Sunday"].copy()

	# Store type is needed to price shedding (Pak 'n Save vs other).
	loc = pd.read_csv(LOCATIONS_FILE)[["Supermarket", "Type"]]
	return test.merge(loc, on="Supermarket")


def load_estimate(path):
	df = pd.read_csv(path)

	# Wide format: a Supermarket column plus Monday..Saturday estimate columns.
	if all(day in df.columns for day in WEEKDAYS[:-1]):
		est = df[["Supermarket"] + WEEKDAYS[:-1]].set_index("Supermarket").astype(float)
		return est

	# Long format: Supermarket, Weekday, and one numeric value column
	# (e.g. Pallet Size). Sunday rows are dropped.
	value_cols = [c for c in df.columns if c not in ["Supermarket", "Weekday", "Type", "Total"]]
	if "Weekday" in df.columns and value_cols:
		est = df[df["Weekday"] != "Sunday"].pivot_table(
			index="Supermarket",
			columns="Weekday",
			values=value_cols[0],
			aggfunc="mean",
		)
		return est.reindex(columns=WEEKDAYS[:-1]).fillna(0.0)

	raise ValueError(
		f"{os.path.basename(path)}: expected Supermarket + Monday..Saturday columns "
		"or Supermarket + Weekday + value columns"
	)


def compare_method(test, est):
	per_day = []
	delivered_total = 0
	required_total = 0
	estimated_total = 0
	dollar_cost_total = 0
	max_shed_pct = 0

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
		shortfall = (-error).clip(lower=0)
		excess = error.clip(lower=0)

		# Under-delivery is covered by wet-leasing the short pallets, or by
		# shedding the store if that is cheaper for that store-day.
		wet_lease_cost = shortfall * UNDER_COST_PER_PALLET
		shed_cost = np.where(
			sub["Type"] == "Pak 'n Save", SHED_COST_PAKNSAVE, SHED_COST_OTHER
		) * (shortfall > 0)
		use_shed = (shed_cost < wet_lease_cost) & (shortfall > 0)
		recourse_cost = np.where(use_shed, shed_cost, wet_lease_cost)

		dollar_cost_total += (excess * OVER_COST_PER_PALLET + recourse_cost).sum()
		max_shed_pct = max(max_shed_pct, 100 * use_shed.sum() / len(sub))

		day_pct = 100 * sub["Delivered"].sum() / sub["Demand"].sum() if sub["Demand"].sum() else np.nan
		per_day.append((sub["Date"].iloc[0], day_pct))

	overall_pct = 100 * delivered_total / required_total if required_total else np.nan
	excess_pct = 100 * (estimated_total - delivered_total) / required_total if required_total else np.nan

	per_day_df = pd.DataFrame(per_day, columns=["Date", "Pct"])
	worst = per_day_df.sort_values("Pct").iloc[0]
	return (
		overall_pct,
		excess_pct,
		dollar_cost_total,
		max_shed_pct,
		worst["Pct"],
		worst["Date"].strftime("%d/%m"),
		required_total,
	)


def main():
	os.makedirs(ESTIMATES_DIR, exist_ok=True)

	test = load_test_demand()

	files = sorted(glob.glob(os.path.join(ESTIMATES_DIR, "*.csv")))
	files = [f for f in files if os.path.basename(f).lower() != "test_demand.csv"]
	if not files:
		print(f"No estimate CSVs found in '{ESTIMATES_DIR}'.")
		return

	rows = []
	for path in files:
		name = os.path.basename(path).removesuffix(".csv")
		print(f"Comparing {name} ...")
		try:
			est = load_estimate(path)
			overall, excess, cost, max_shed, worst_pct, worst_date, required = compare_method(test, est)
		except (ValueError, KeyError) as err:
			print(f"  SKIPPED: {err}")
			continue
		rows.append(
			{
				"Method": name,
				"Error cost ($)": round(cost),
				"Delivered %": round(overall, 1),
				"Excess %": round(excess, 1),
				"Max shed %": round(max_shed, 1),
				"Worst day %": round(worst_pct, 1),
				"Worst day": worst_date,
				"Test pallets (required)": int(required),
			}
		)

	result = pd.DataFrame(rows).sort_values("Error cost ($)").reset_index(drop=True)
	result.index += 1

	print("\nRanking by estimated error cost ($) - lowest is best:\n")
	print(result.to_string(index=True))


if __name__ == "__main__":
	main()