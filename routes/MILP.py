"""
MILP for route selection - Part I, Section 3

Takes the route pool CSV and selects the cheapest combination of routes
that covers every store. Same structure as the set partitioning problem
in the coursebook: each store is an "object", each candidate route is a
"set", and the model chooses which sets to select so every object is
covered exactly once.

Two versions are solved:

BASE - every store must be covered by a selected route, no skipping
allowed. Standard Ax=e partitioning.

FUEL REDUCTION - same model, but every store also has the option of
being skipped instead of covered, at a cost equal to its penalty ($1500
for Pak'n Save, $800 for everything else), capped at 20% of stores.
This is modelled as a singleton set containing only that store, so the
problem stays a genuine partition (every store covered by exactly one
set) rather than needing separate logic bolted on for skipping.

Both versions also allow a route to be run wet-leased through Linfox
instead of the in-house fleet ($1400 per 2hr block, uncapped), which
becomes relevant once the 40 available in-house truck-shifts (20 trucks
x 2 shifts/day) aren't enough to cover everything.
"""

import pandas as pd
import pulp
import math

ROUTE_POOL_PATH = "route_pool_all_days.csv"
DEMAND_PATH = "8week-estimated_demand.csv"

TRUCKS = 20
SHIFTS_PER_TRUCK = 2
MAX_TRUCK_SHIFTS = TRUCKS * SHIFTS_PER_TRUCK  # 40 total route-uses/day on own fleet

SHED_LIMIT_FRACTION = 0.20  # at most 20% of stores can be shed


def wet_lease_cost(total_min):
    """$1400 per 2-hour block, charged in whole blocks."""
    hours = total_min / 60.0
    blocks = math.ceil(hours / 2.0)
    return 1400 * blocks


def load_store_penalties(demand_path):
    """Shed penalty per store: $1500 for Pak'n Save, $800 for everything else."""
    demand = pd.read_csv(demand_path)
    penalties = {}
    for _, row in demand.iterrows():
        store = row['Supermarket']
        penalties[store] = 1500 if row['Type'] == "Pak 'n Save" else 800
    return penalties


def solve_day(day_routes, store_penalties, allow_shedding, day_name):
    """
    day_routes: DataFrame of candidate routes for one weekday
                (columns: RouteID, Stores, TotalCost, TotalMin, ...)
    allow_shedding: if True, adds the shed-singleton option per store
                    and the 20% GUB limit (fuel-reduction variant).
                    If False, every store must be covered (base model).
    """
    stores_today = sorted(set(
        s for stores in day_routes['Stores'] for s in stores.split(',')
    ))

    prob = pulp.LpProblem(f"RouteSelection_{day_name}", pulp.LpMinimize)

    # One binary variable per route for "run with in-house fleet" (x) and
    # "run wet-leased" (y). z is the "skip this store" variable - only
    # created when shedding is enabled.
    x = {rid: pulp.LpVariable(f"x_{rid}", cat="Binary") for rid in day_routes['RouteID']}
    y = {rid: pulp.LpVariable(f"y_{rid}", cat="Binary") for rid in day_routes['RouteID']}
    z = {}
    if allow_shedding:
        z = {s: pulp.LpVariable(f"z_{s.replace(' ', '_').replace(chr(39),'')}", cat="Binary")
             for s in stores_today}

    route_cost = dict(zip(day_routes['RouteID'], day_routes['TotalCost']))
    route_wet_cost = {rid: wet_lease_cost(tm) for rid, tm in
                       zip(day_routes['RouteID'], day_routes['TotalMin'])}
    route_stores = dict(zip(day_routes['RouteID'],
                             day_routes['Stores'].apply(lambda s: s.split(','))))

    # Objective: sum of costs for every variable switched on - route
    # costs, wet-lease costs, and skip penalties.
    prob += (
        pulp.lpSum(route_cost[rid] * x[rid] for rid in x)
        + pulp.lpSum(route_wet_cost[rid] * y[rid] for rid in y)
        + pulp.lpSum(store_penalties[s] * z[s] for s in z)
    )

    # Coverage: every store must be served exactly once - either by a
    # selected route (in-house or wet-leased) or by being skipped, never
    # zero times and never twice.
    for s in stores_today:
        covering_routes = [rid for rid, stlist in route_stores.items() if s in stlist]
        shed_term = z[s] if allow_shedding else 0
        prob += (
            pulp.lpSum(x[rid] + y[rid] for rid in covering_routes) + shed_term == 1,
            f"cover_{s.replace(' ', '_').replace(chr(39),'')}"
        )

    # Fleet limit: only 40 in-house truck-shifts available (20 trucks,
    # 2 shifts each).
    prob += pulp.lpSum(x.values()) <= MAX_TRUCK_SHIFTS, "fleet_limit"

    # A route cannot be both in-house and wet-leased at the same time.
    for rid in x:
        prob += x[rid] + y[rid] <= 1, f"no_double_{rid}"

    # Fuel reduction limit: no more than 20% of stores can be skipped.
    if allow_shedding:
        prob += pulp.lpSum(z.values()) <= SHED_LIMIT_FRACTION * len(stores_today), "shed_limit"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    used_routes = [rid for rid in x if x[rid].value() and x[rid].value() > 0.5]
    wet_routes = [rid for rid in y if y[rid].value() and y[rid].value() > 0.5]
    shed_stores = [s for s in z if z[s].value() and z[s].value() > 0.5]

    return {
        'day': day_name,
        'status': status,
        'total_cost': pulp.value(prob.objective),
        'n_own_routes': len(used_routes),
        'n_wet_routes': len(wet_routes),
        'n_shed_stores': len(shed_stores),
        'own_routes': used_routes,
        'wet_routes': wet_routes,
        'shed_stores': shed_stores,
        'n_stores_total': len(stores_today),
    }


def main():
    pool = pd.read_csv(ROUTE_POOL_PATH)
    penalties = load_store_penalties(DEMAND_PATH)

    results_base, results_fuel = [], []

    for day in pool['Weekday'].unique():
        day_routes = pool[pool['Weekday'] == day]

        base = solve_day(day_routes, penalties, allow_shedding=False, day_name=day)
        results_base.append(base)
        print(f"[BASE]  {day}: status={base['status']}, cost=${base['total_cost']:.2f}, "
              f"own-fleet routes={base['n_own_routes']}, wet-lease routes={base['n_wet_routes']}")

        fuel = solve_day(day_routes, penalties, allow_shedding=True, day_name=day)
        results_fuel.append(fuel)
        print(f"[FUEL]  {day}: status={fuel['status']}, cost=${fuel['total_cost']:.2f}, "
              f"own-fleet routes={fuel['n_own_routes']}, wet-lease routes={fuel['n_wet_routes']}, "
              f"shed={fuel['n_shed_stores']}/{fuel['n_stores_total']} stores")
        print()

    base_df = pd.DataFrame(results_base)
    fuel_df = pd.DataFrame(results_fuel)
    base_df.to_csv('milp_results_base.csv', index=False)
    fuel_df.to_csv('milp_results_fuel_reduction.csv', index=False)

    print(f"\nTotal weekly cost (base, no shedding): ${base_df['total_cost'].sum():.2f}")
    print(f"Total weekly cost (fuel reduction):     ${fuel_df['total_cost'].sum():.2f}")
    print(f"Weekly savings from shedding option:    ${base_df['total_cost'].sum() - fuel_df['total_cost'].sum():.2f}")

    return base_df, fuel_df


if __name__ == '__main__':
    main()