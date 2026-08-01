import os
import numpy as np
import pandas as pd

DEMAND_FILE = "FoodstuffsDemand2026.csv"
LOCATIONS_FILE = "FoodstuffsLocations.csv"
OUTPUT_DIR = "estimations"

OUTLIER_THRESHOLD = 20
PUBLIC_HOLIDAYS = ["1/06/2026"]
TRAIN_END = pd.Timestamp("2026-06-14")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	locations = pd.read_csv(LOCATIONS_FILE)[["Supermarket", "Type"]]

	demand = pd.read_csv(DEMAND_FILE)
	long_data = demand.melt(
		id_vars="Supermarket",
		value_vars=demand.columns[1:],
		var_name="Date",
		value_name="Demand",
	)
	long_data["Date"] = pd.to_datetime(long_data["Date"], dayfirst=True)
	long_data["DayOfWeek"] = long_data["Date"].dt.day_name()
	long_data["IsHoliday"] = long_data["Date"].dt.strftime("%d/%m/%Y").isin(PUBLIC_HOLIDAYS)
	long_data.loc[long_data["Demand"] >= OUTLIER_THRESHOLD, "Demand"] = np.nan

	train = long_data[long_data["Date"] <= TRAIN_END]
	test = long_data[long_data["Date"] > TRAIN_END].copy()

	train_used = train[~train["IsHoliday"] & (train["DayOfWeek"] != "Sunday")]
	train_used["Demand"] = train_used.groupby(["Supermarket", "DayOfWeek"])["Demand"].transform(
		lambda s: s.fillna(s.median())
	)

	stats = train_used.groupby(["Supermarket", "DayOfWeek"])["Demand"]
	means = stats.mean().unstack(fill_value=0).reindex(columns=WEEKDAYS).fillna(0.0)
	stds = stats.std().unstack(fill_value=0).reindex(columns=WEEKDAYS).fillna(0.0)
	est = np.ceil(means + stds).astype(int)
	est = est.merge(locations, on="Supermarket").set_index("Supermarket")
	est["Total"] = est[WEEKDAYS[:-1]].sum(axis=1)

	est.to_csv(os.path.join(OUTPUT_DIR, "6week-estimated_demand.csv"))

	test = test.merge(locations, on="Supermarket")
	test.to_csv(os.path.join(OUTPUT_DIR, "test_demand.csv"), index=False)

	test_days = test[test["DayOfWeek"] != "Sunday"].copy()
	test_days["Estimate"] = test_days.apply(
		lambda r: est.loc[r["Supermarket"], r["DayOfWeek"]], axis=1
	)
	coverage = (test_days["Demand"] <= test_days["Estimate"]).mean() * 100
	mae = (test_days["Demand"] - test_days["Estimate"]).abs().mean()

	print("Train weeks:", train["Date"].min().date(), "to", train["Date"].max().date())
	print("Test weeks:", test["Date"].min().date(), "to", test["Date"].max().date())
	print("\nEstimated pallets per store per weekday:\n")
	print(est.to_string())
	print("\nTotal estimated pallets per day:\n")
	print(est[WEEKDAYS[:-1]].sum().to_string())
	print(f"\nTest coverage (actual <= estimate): {coverage:.1f}%")
	print(f"Mean absolute error on test: {mae:.2f} pallets")


if __name__ == "__main__":
	main()
