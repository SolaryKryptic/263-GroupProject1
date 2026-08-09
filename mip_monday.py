"""
mip_monday.py

MILP route selection for Monday using v3 pool.
Uses scipy.optimize.milp (HiGHS).
"""

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

DEMAND_FILE = "estimations/0.5ayush6week-estimated_demand.csv"
POOL_FILE = "v3-routes/pool_Monday.csv"
LOCATIONS_FILE = "FoodstuffsLocations.csv"

# Problem constants
TRUCK_CAPACITY = 16
BASE_TRIP_CAP_MIN = 210
BASE_COST_PER_HOUR = 220.0
OVERTIME_COST_PER_HOUR = 310.0
WET_LEASE_BLOCK_COST = 1400.0
WET_LEASE_BLOCK_MIN = 120
SHED_COST_PAKNSAVE = 1500.0
SHED_COST_OTHER = 800.0
MAX_SHED_PCT = 0.2
MAX_OWNED_TRIPS = 40  # 20 trucks * 2 shifts


def load_demand():
    est = pd.read_csv(DEMAND_FILE)
    est = est.set_index("Supermarket")
    monday = est["Monday"].astype(int)
    return monday[monday > 0]


def load_pool():
    df = pd.read_csv(POOL_FILE)
    return df


def load_locations():
    df = pd.read_csv(LOCATIONS_FILE)
    return df.set_index("Supermarket")["Type"].to_dict()


def build_wet_cost(total_min):
    """Wet-lease cost: ceil(total_min / 120) * 1400"""
    return np.ceil(total_min / 120) * 1400


def main():
    # Load data
    demand = load_demand()
    pool = load_pool()
    store_types = load_locations()

    stores = list(demand.index)
    store_to_idx = {s: i for i, s in enumerate(stores)}
    n_stores = len(stores)
    n_routes = len(pool)

    # Shed cost per store
    shed_cost = np.array([
        1500.0 if store_types.get(s) == "Pak 'n Save" else 800.0
        for s in stores
    ])

    # Build incidence matrix A (stores x routes)
    row_indices = []
    col_indices = []
    for r_idx, row in pool.iterrows():
        store_names = row["Stores"].split(",")
        for s in store_names:
            if s in store_to_idx:
                row_indices.append(store_to_idx[s])
                col_indices.append(r_idx)
    A = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)),
                   shape=(n_stores, n_routes))

    # Route costs
    total_cost = pool["TotalCost"].values.astype(float)
    total_min = pool["TotalMin"].values.astype(float)
    wet_cost = np.ceil(total_min / 120) * 1400.0

    # Decision variables: [z (owned), y (wet), s (shed)]
    n = n_routes * 2 + n_stores

    # Objective: min sum(TotalCost * z) + sum(WetCost * y) + sum(ShedCost * s)
    c = np.concatenate([total_cost, wet_cost, shed_cost])

    # Bounds: binary variables
    bounds = Bounds(0, 1)
    integrality = np.ones(n)

    # Constraints
    # 1. Coverage: A * (z + y) + s = 1  ->  A*z + A*y + I*s = 1
    A1 = csr_matrix(np.hstack([A.toarray(), A.toarray(), np.eye(n_stores)]))
    coverage_lb = np.ones(n_stores)
    coverage_ub = np.ones(n_stores)

    # 2. Owned fleet: sum(z) <= 40
    A2 = np.hstack([np.ones(n_routes), np.zeros(n_routes), np.zeros(n_stores)])
    fleet_ub = np.array([40.0])

    # 3. z + y <= 1 for each route
    A3 = csr_matrix(np.hstack([np.eye(n_routes), np.eye(n_routes), np.zeros((n_routes, n_stores))]))
    route_ub = np.ones(n_routes)

    # 4. Shed cap: sum(s) <= 0.2 * n_stores
    A4 = np.hstack([np.zeros(n_routes), np.zeros(n_routes), np.ones(n_stores)])
    shed_ub = np.array([int(MAX_SHED_PCT * n_stores)])

    # Stack all constraints
    A_all = csr_matrix(np.vstack([A1.toarray(), A2, A3.toarray(), A4]))
    lb = np.concatenate([coverage_lb, [-np.inf], route_ub * 0, [-np.inf]])
    ub = np.concatenate([coverage_ub, fleet_ub, route_ub, shed_ub])

    print(f"Problem size: {n} vars, {A_all.shape[0]} constraints")
    print(f"Stores: {n_stores}, Routes: {n_routes}")
    print(f"Shed cap: {shed_ub[0]} stores")

    # Solve
    constraints = LinearConstraint(A_all, lb, ub)
    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality,
                  options={'time_limit': 60, 'disp': True})

    if result.success:
        x = result.x
        z = x[:n_routes]
        y = x[n_routes:2*n_routes]
        s = x[2*n_routes:]

        selected_owned = np.where(z > 0.5)[0]
        selected_wet = np.where(y > 0.5)[0]
        shed = np.where(s > 0.5)[0]

        owned_cost = total_cost[selected_owned].sum()
        wet_cost_sum = wet_cost[selected_wet].sum()
        shed_cost_sum = shed_cost[shed].sum()
        total = owned_cost + wet_cost_sum + shed_cost_sum

        print(f"\n=== SOLUTION ===")
        print(f"Owned trips: {len(selected_owned)}")
        print(f"Wet-leased trips: {len(selected_wet)}")
        print(f"Shed stores: {len(shed)}")
        print(f"Owned cost: ${owned_cost:.2f}")
        print(f"Wet cost: ${wet_cost_sum:.2f}")
        print(f"Shed cost: ${shed_cost_sum:.2f}")
        print(f"Total cost: ${total:.2f}")

        # Save selected routes
        out_rows = []
        for idx in selected_owned:
            row = pool.iloc[idx].copy()
            row["Mode"] = "Owned"
            out_rows.append(row)
        for idx in selected_wet:
            row = pool.iloc[idx].copy()
            row["Mode"] = "Wet-Leased"
            out_rows.append(row)
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv("monday_solution.csv", index=False)
        print("Saved monday_solution.csv")

        if len(shed) > 0:
            shed_names = [stores[i] for i in shed]
            print(f"Shed stores: {shed_names}")
    else:
        print(f"Solver failed: {result.message}")


if __name__ == "__main__":
    main()