"""
mip_weekly.py

MILP route selection for all weekdays using v3 pools.
"""

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

DEMAND_FILE = "estimations/0.5ayush6week-estimated_demand.csv"
POOL_DIR = "v3"
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
MAX_OWNED_TRIPS = 40
SATURDAY_POOL_LIMIT = 20000  # 20k cheapest routes for Saturday


def build_wet_cost(total_min):
    """Wet-lease cost: ceil(total_min / 120) * 1400"""
    return np.ceil(total_min / 120) * 1400


def solve_weekday(weekday, pool, demand, store_types):
    """Solve MILP for one weekday."""
    stores = list(demand.index)
    store_to_idx = {s: i for i, s in enumerate(stores)}
    n_stores = len(stores)
    n_routes = len(pool)

    # Shed cost per store
    shed_cost = np.array([
        SHED_COST_PAKNSAVE if store_types.get(s) == "Pak 'n Save" else SHED_COST_OTHER
        for s in stores
    ])

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

    # Variables: [z (owned), y (wet), s (shed)]
    n_routes = len(pool)
    n_stores = len(stores)
    n = n_routes * 2 + n_stores

    # Objective
    c = np.concatenate([total_cost, build_wet_cost(total_min), shed_cost])

    bounds = Bounds(0, 1)
    integrality = np.ones(n)

    # Coverage: A * (z + y) + s = 1
    A1 = csr_matrix(np.hstack([A.toarray(), A.toarray(), np.eye(n_stores)]))
    coverage_lb = np.ones(n_stores)
    coverage_ub = np.ones(n_stores)

    # Fleet: sum(z) <= 40
    A2 = np.hstack([np.ones(n_routes), np.zeros(n_routes), np.zeros(n_stores)])
    fleet_ub = np.array([40.0])

    # Route usage: z + y <= 1
    A3 = csr_matrix(np.hstack([np.eye(n_routes), np.eye(n_routes), np.zeros((n_routes, n_stores))]))
    route_ub = np.ones(n_routes)

    # Shed cap: sum(s) <= 11
    A4 = np.hstack([np.zeros(n_routes), np.zeros(n_routes), np.ones(n_stores)])
    shed_ub = np.array([int(MAX_SHED_PCT * n_stores)])

    # Stack constraints
    A_all = csr_matrix(np.vstack([A1.toarray(), A2, A3.toarray(), A4]))
    lb = np.concatenate([coverage_lb, [-np.inf], np.zeros(n_routes), [-np.inf]])
    ub = np.concatenate([coverage_ub, np.array([40.0]), np.ones(n_routes), [shed_ub[0]]])

    constraints = LinearConstraint(A_all, lb, ub)

    result = milp(c=c, constraints=constraints, bounds=Bounds(0, 1), integrality=integrality,
                  options={'time_limit': 120, 'disp': False})

    if not result.success:
        return None

    x = result.x
    z = x[:n_routes]
    y = x[n_routes:2*n_routes]
    s = x[2*n_routes:]

    selected_owned = np.where(z > 0.5)[0]
    selected_wet = np.where(y > 0.5)[0]
    shed = np.where(s > 0.5)[0]

    owned_cost = total_cost[selected_owned].sum()
    wet_cost_sum = build_wet_cost(total_min)[selected_wet].sum()
    shed_cost_sum = shed_cost[shed].sum()
    total = owned_cost + wet_cost_sum + shed_cost_sum

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
    shed_names = [stores[i] for i in shed]

    return {
        "weekday": weekday,
        "solution": out_df,
        "owned_cost": owned_cost,
        "wet_cost": wet_cost_sum,
        "shed_cost": shed_cost_sum,
        "total_cost": total,
        "n_owned": len(selected_owned),
        "n_wet": len(selected_wet),
        "n_shed": len(shed),
        "shed_stores": shed_names,
        "total_pallets": out_df["Pallets"].sum() if len(out_df) else 0
    }


def main():
    # Load data
    demand_df = pd.read_csv(DEMAND_FILE).set_index("Supermarket")
    store_types = pd.read_csv(LOCATIONS_FILE).set_index("Supermarket")["Type"].to_dict()

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    results = []

    for weekday in weekdays:
        print(f"\n=== {weekday} ===")
        pool = pd.read_csv(f"{POOL_DIR}/pool_{weekday}.csv")
        
        # Limit Saturday pool
        if weekday == "Saturday" and len(pool) > SATURDAY_POOL_LIMIT:
            pool = pool.nsmallest(SATURDAY_POOL_LIMIT, "TotalCost").reset_index(drop=True)
            print(f"  Limited Saturday pool to {SATURDAY_POOL_LIMIT} routes")

        demand = demand_df[weekday].astype(int)
        demand = demand[demand > 0]

        result = solve_weekday(weekday, pool, demand, store_types)
        
        if result:
            results.append(result)
            print(f"  Owned: {result['n_owned']}, Wet: {result['n_wet']}, Shed: {result['n_shed']}")
            print(f"  Cost: ${result['total_cost']:.2f} (Owned: ${result['owned_cost']:.2f}, Wet: ${result['wet_cost']:.2f}, Shed: ${result['shed_cost']:.2f})")
            print(f"  Pallets: {result['total_pallets']}")
            if result['shed_stores']:
                print(f"  Shed: {result['shed_stores']}")
            
            # Save solution
            result["solution"].to_csv(f"{weekday}_solution.csv", index=False)
        else:
            print("  FAILED")

    # Summary
    print("\n=== WEEKLY SUMMARY ===")
    total_weekly = sum(r["total_cost"] for r in results)
    print(f"Weekly base cost: ${total_weekly:.2f}")
    for r in results:
        print(f"  {r['weekday']:10s} ${r['total_cost']:>10.2f} | Owned: {r['n_owned']} | Wet: {r['n_wet']} | Shed: {r['n_shed']} | Pallets: {r['total_pallets']}")

    pd.DataFrame([{
        "Weekday": r["weekday"],
        "TotalCost": r["total_cost"],
        "OwnedCost": r["owned_cost"],
        "WetCost": r["wet_cost"],
        "ShedCost": r["shed_cost"],
        "OwnedTrips": r["n_owned"],
        "WetTrips": r["n_wet"],
        "ShedStores": r["n_shed"],
        "TotalPallets": r["total_pallets"],
        "ShedList": ", ".join(r["shed_stores"])
    } for r in results]).to_csv("weekly_summary.csv", index=False)
    print("Saved weekly_summary.csv")


if __name__ == "__main__":
    main()