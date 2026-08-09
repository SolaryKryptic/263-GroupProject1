"""
mip_weekly_no_shed.py

MILP route selection for all weekdays using v3 pools with NO SHEDDING.
"""

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

DEMAND_FILE = "estimations/0.5ayush6week-estimated_demand.csv"
POOL_DIR = "v3-routes"
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
MAX_OWNED_TRIPS = 40
SATURDAY_POOL_LIMIT = 20000


def build_wet_cost(total_min):
    """Wet-lease cost: ceil(total_min / 120) * 1400"""
    return np.ceil(total_min / 120) * 1400


def solve_weekday(weekday, pool, demand, store_types):
    """Solve MILP for one weekday — NO SHEDDING."""
    stores = list(demand.index)
    store_to_idx = {s: i for i, s in enumerate(stores)}
    n_stores = len(stores)
    n_routes = len(pool)

    # Build incidence matrix A (stores x routes)
    row_indices = []
    col_indices = []
    for r_idx, row in pool.iterrows():
        store_names = row["Stores"].split(",")
        for s in store_names:
            s_clean = s.strip()
            if s_clean in store_to_idx:
                row_indices.append(store_to_idx[s_clean])
                col_indices.append(r_idx)
    A = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)),
                   shape=(n_stores, n_routes))

    # Route costs
    total_cost = pool["TotalCost"].values.astype(float)
    total_min = pool["TotalMin"].values.astype(float)
    wet_cost = np.ceil(total_min / 120) * 1400.0

    n_routes = len(pool)
    n_stores = len(stores)
    n = n_routes * 2  # z (owned), y (wet)

    # Objective
    c = np.concatenate([pool["TotalCost"].values.astype(float), np.ceil(pool["TotalMin"].values.astype(float) / 120) * 1400.0])

    bounds = Bounds(0, 1)
    integrality = np.ones(n)

    # Constraints
    # 1. Coverage: A * (z + y) = 1
    A1 = csr_matrix(np.hstack([A.toarray(), A.toarray()]))
    coverage_lb = np.ones(n_stores)
    coverage_ub = np.ones(n_stores)

    # 2. Owned fleet: sum(z) <= 40
    A2 = np.hstack([np.ones(n_routes), np.zeros(n_routes)])
    fleet_ub = np.array([40.0])

    # 3. z + y <= 1
    A3 = csr_matrix(np.hstack([np.eye(n_routes), np.eye(n_routes)]))
    route_ub = np.ones(n_routes)

    # Stack constraints
    A_all = csr_matrix(np.vstack([A1.toarray(), A2, A3.toarray()]))
    lb = np.concatenate([np.ones(n_stores), [-np.inf], np.zeros(n_routes)])
    ub = np.concatenate([np.ones(n_stores), np.array([40.0]), np.ones(n_routes)])

    constraints = LinearConstraint(A_all, lb, ub)

    result = milp(c=c, constraints=constraints, bounds=Bounds(0, 1), integrality=np.ones(n),
                  options={'time_limit': 120, 'disp': False})

    if not result.success:
        return None

    x = result.x
    z = x[:n_routes]
    y = x[n_routes:2*len(pool)]

    selected_owned = np.where(z > 0.5)[0]
    selected_wet = np.where(y > 0.5)[0]

    total_cost_vals = pool["TotalCost"].values.astype(float)
    wet_cost = np.ceil(pool["TotalMin"].values.astype(float) / 120) * 1400.0

    owned_cost = total_cost_vals[selected_owned].sum()
    wet_cost_sum = wet_cost[selected_wet].sum()
    total = owned_cost + wet_cost_sum

    # Build solution dataframe
    out_rows = []
    for idx in selected_owned:
        row = pool.iloc[idx].copy()
        row["Mode"] = "Owned"
        out_rows.append(row)
    for idx in selected_wet:
        row = pool.iloc[idx].copy()
        row["Mode"] = "Wet-Leased"
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame()

    return {
        "weekday": weekday,
        "solution": out_df,
        "owned_cost": float(owned_cost),
        "wet_cost": float(wet_cost_sum),
        "total_cost": float(total),
        "n_owned": len(selected_owned),
        "n_wet": len(selected_wet),
        "total_pallets": int(out_df["Pallets"].sum()) if len(out_df) else 0
    }


def main():
    demand_df = pd.read_csv("estimations/0.5ayush6week-estimated_demand.csv").set_index("Supermarket")
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    results = []

    for weekday in weekdays:
        print(f"\n=== {weekday} ===")
        pool = pd.read_csv(f"{POOL_DIR}/pool_{weekday}.csv")
        
        if weekday == "Saturday" and len(pool) > 20000:
            pool = pool.nsmallest(20000, "TotalCost").reset_index(drop=True)
            print(f"  Limited Saturday pool to 20000 routes")

        demand = pd.read_csv("estimations/0.5ayush6week-estimated_demand.csv").set_index("Supermarket")[weekday].astype(int)
        demand = demand[demand > 0]

        result = solve_weekday(weekday, pool, demand, None)
        
        if result:
            results.append(result)
            print(f"  Owned: {result['n_owned']}, Wet: {result['n_wet']}")
            print(f"  Cost: ${result['total_cost']:.2f} (Owned: ${result['owned_cost']:.2f}, Wet: ${result['wet_cost']:.2f})")
            print(f"  Pallets: {result['total_pallets']}")
            
            result["solution"].to_csv(f"{weekday}_solution_no_shed.csv", index=False)
        else:
            print("  FAILED")

    # Summary
    print("\n=== WEEKLY SUMMARY (NO SHED) ===")
    total_weekly = sum(r["total_cost"] for r in results)
    print(f"Weekly base cost: ${total_weekly:.2f}")
    for r in results:
        print(f"  {r['weekday']:10s} ${r['total_cost']:>10.2f} | Owned: {r['n_owned']} | Wet: {r['n_wet']} | Pallets: {r['total_pallets']}")

    pd.DataFrame([{
        "Weekday": r["weekday"],
        "TotalCost": r["total_cost"],
        "OwnedCost": r["owned_cost"],
        "WetCost": r["wet_cost"],
        "OwnedTrips": r["n_owned"],
        "WetTrips": r["n_wet"],
        "TotalPallets": r["total_pallets"]
    } for r in results]).to_csv("weekly_summary_no_shed.csv", index=False)
    print("Saved weekly_summary_no_shed.csv")


if __name__ == "__main__":
    main()