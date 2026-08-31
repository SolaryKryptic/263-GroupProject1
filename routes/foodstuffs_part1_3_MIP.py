import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ------------------------------------------------------------
# Foodstuffs NZ - Part I.3 Mixed Integer Programming
# ------------------------------------------------------------

# Input files
EST_FILE = "ayush_estimates.csv"
ROUTE_FILE = "routes_Monday.csv"

est = pd.read_csv(EST_FILE)
routes = pd.read_csv(ROUTE_FILE)

stores = est["Supermarket"].tolist()
n_stores = len(stores)
n_routes = len(routes)
store_index = {s: i for i, s in enumerate(stores)}

# A[i,j] = 1 if route j visits store i
A = np.zeros((n_stores, n_routes))
for j, stops in enumerate(routes["stops"]):
    for store in stops.split(";"):
        if store not in store_index:
            raise ValueError(f"Store in route file not found in estimates: {store}")
        A[store_index[store], j] = 1

route_cost = routes["cost"].to_numpy(dtype=float)

# ============================================================
# 1. BASE MIP
# ============================================================
#
# x_j = 1 if route j is selected
# w   = number of wet-leased routes/truck slots required
#
# Minimise:
#       sum_j cost_j*x_j + 1400*w
#
# Subject to:
#       every store is served exactly once
#       number of Foodstuffs routes <= 40 + wet leases
#       x_j binary, w integer >= 0
#
# The 40 comes from 20 trucks x 2 shifts per day.
# Every Monday route supplied is <= 3.5 hours and <= 16 pallets.

c = np.r_[route_cost, 1400.0]

# Store coverage: A*x = 1
coverage = np.hstack([A, np.zeros((n_stores, 1))])

# Fleet capacity: sum(x) - w <= 40
fleet = np.r_[np.ones(n_routes), -1.0].reshape(1, -1)

constraints = [
    LinearConstraint(coverage, np.ones(n_stores), np.ones(n_stores)),
    LinearConstraint(fleet, -np.inf, 40)
]

integrality = np.ones(n_routes + 1)
lower = np.zeros(n_routes + 1)
upper = np.r_[np.ones(n_routes), np.inf]

result = milp(
    c=c,
    integrality=integrality,
    bounds=Bounds(lower, upper),
    constraints=constraints
)

if not result.success:
    raise RuntimeError(result.message)

selected = np.where(result.x[:n_routes] > 0.5)[0]
wet_leases = round(result.x[-1])

print("\nBASE MODEL")
print("Optimal cost: ${:,.2f}".format(result.fun))
print("Routes selected:", len(selected))
print("Wet leases required:", wet_leases)
print("Total pallets:", int(routes.iloc[selected]["total_pallets"].sum()))
print("Total truck-hours:", routes.iloc[selected]["total_duration_sec"].sum() / 3600)

print("\nSelected routes:")
print(
    routes.iloc[selected][
        ["route_id", "stops", "num_stops",
         "total_pallets", "total_duration_sec", "cost"]
    ].to_string(index=False)
)

# ============================================================
# 2. FUEL-REDUCTION MODEL
# ============================================================
#
# s_i = 1 if store i is skipped ("demand shed")
#
# A*x + s = 1
# at most 20% of stores can be shed
#
# Penalty:
#   Pak 'n Save = $1500
#   all other stores = $800
#
# Objective:
#   route operating costs + wet-lease costs + shedding penalties

shed_cost = np.array([
    1500.0 if t == "Pak 'n Save" else 800.0
    for t in est["Type"]
])

c_fuel = np.r_[route_cost, 1400.0, shed_cost]

# Variables are [x routes, w wet leases, s stores]
coverage_fuel = np.hstack([
    A,
    np.zeros((n_stores, 1)),
    np.eye(n_stores)
])

fleet_fuel = np.r_[
    np.ones(n_routes),
    -1.0,
    np.zeros(n_stores)
].reshape(1, -1)

shed_limit = np.r_[
    np.zeros(n_routes + 1),
    np.ones(n_stores)
].reshape(1, -1)

constraints_fuel = [
    LinearConstraint(
        coverage_fuel,
        np.ones(n_stores),
        np.ones(n_stores)
    ),
    LinearConstraint(fleet_fuel, -np.inf, 40),
    LinearConstraint(shed_limit, -np.inf, 0.20 * n_stores)
]

integrality_fuel = np.ones(n_routes + 1 + n_stores)
lower_fuel = np.zeros(n_routes + 1 + n_stores)
upper_fuel = np.r_[
    np.ones(n_routes),
    np.inf,
    np.ones(n_stores)
]

result_fuel = milp(
    c=c_fuel,
    integrality=integrality_fuel,
    bounds=Bounds(lower_fuel, upper_fuel),
    constraints=constraints_fuel
)

if not result_fuel.success:
    raise RuntimeError(result_fuel.message)

selected_fuel = np.where(result_fuel.x[:n_routes] > 0.5)[0]
shed = np.where(result_fuel.x[n_routes + 1:] > 0.5)[0]
wet_leases_fuel = round(result_fuel.x[n_routes])

print("\nFUEL-REDUCTION MODEL")
print("Optimal cost: ${:,.2f}".format(result_fuel.fun))
print("Routes selected:", len(selected_fuel))
print("Stores shed:", len(shed))
print("Wet leases required:", wet_leases_fuel)
print("Shed penalty: ${:,.2f}".format(shed_cost[shed].sum()))

if len(shed):
    print("\nStores shed:")
    print(est.iloc[shed][["Supermarket", "Type", "Monday"]].to_string(index=False))
else:
    print("\nNo stores are shed because the demand-shedding penalties exceed "
          "the savings from reducing routes.")

print("\nSelected fuel-model routes:")
print(
    routes.iloc[selected_fuel][
        ["route_id", "stops", "num_stops",
         "total_pallets", "total_duration_sec", "cost"]
    ].to_string(index=False)
)
