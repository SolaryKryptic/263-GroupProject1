import pandas as pd



demand = pd.read_csv("FoodstuffsDemand2026.csv")



first_6_weeks = ["Supermarket"] + list(demand.columns[1:43])
demand = demand[first_6_weeks]



demand_long = demand.melt(
    id_vars="Supermarket",
    var_name="Date",
    value_name="Pallets"
)


weekdays = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

demand_long["Weekday"] = (
    demand_long.groupby("Supermarket").cumcount() % 7
).map(dict(enumerate(weekdays)))


demand_long = demand_long[demand_long["Weekday"] != "Sunday"]


estimates = (
    demand_long
    .groupby(["Supermarket", "Weekday"])["Pallets"]
    .median()
    .round()
    .astype(int)
    .unstack()
)



estimates = estimates[
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday"
    ]
]



estimates["Sunday"] = 0



estimates["Type"] = estimates.index.str.extract(
    r"^(Four Square|New World|Pak 'n Save)"
)



estimates["Total"] = (
    estimates[
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    ]
    .sum(axis=1)
)



estimates = estimates.reset_index()



print(estimates)



estimates.to_csv("Median_Demand_Estimates.csv", index=False)

print("\nMedian demand estimates saved successfully!")