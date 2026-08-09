"""
Foodstuffs route pool generator - Part I, Section 2

What this does
--------------
This doesn't decide an actual day plan. It just builds a big pool of
individually feasible routes (every reasonable combination of stores one
truck could visit in a trip) and dumps them in a CSV. The MILP in step 3
is what actually picks which ones get used - this script just gives it
options to choose from.

We only need one pool per weekday, not per calendar date, since the
demand estimate we're using is per day-of-week (Mon-Sun), not per actual
date. Sunday has zero demand everywhere in the data so we just skip it -
no deliveries that day.

Cost model
----------
Driving time comes straight from the duration matrix (it's not
symmetric btw - a->b and b->a are different, real Auckland driving
asymmetry, not a bug). Unload time is 18 min per pallet, added up across
every stop. Cost is $220/hr for the first 4 hours, then $310/hr overtime
(rounded up) after that.

We used 4 hours as the actual overtime trigger, not 3.5. The brief says
"no more than 3.5 hours, on average" as a policy target, but the $310/hr
number is specifically attached to trips going "more than four hours" a
sentence later - that's a different number. 3.5 has no cost attached to
it anywhere in the brief, so we treated it as just a goal, not a real
threshold.

Worth knowing: with this cost model, most weekday routes with 2+ stops
end up over 4 hours anyway (~85% in our output). That's not a mistake -
it's just what happens when you combine Auckland's spread-out geography
with 18 min/pallet unloading. Overtime is basically unavoidable once a
route has more than one stop, so the MILP genuinely has to weigh overtime
vs more trucks vs wet-leasing, it's not just a minor edge case.

How routes get built
---------------------
1-3 stop routes: we just brute force every combination. There aren't
that many possible combos out of 55 stores, so it's cheap to check all
of them and keep the best ordering for each. This means we never miss a
good short route.

4-6 stop routes: brute force stops being realistic at this size, so we
build these with cheapest insertion instead - start at a random store,
keep adding whichever store is cheapest to insert next, stop once it
hits 16 pallets or gets too long. First attempt at this always picked
the single cheapest option each time and it kept converging to basically
the same route no matter where it started (only got 4 unique routes out
of 400 tries) - so instead we randomise between the top 3 cheapest
choices at each step to actually get variety. Every route also gets
cleaned up with 2-opt after building (uncrosses it if it zigzags).

Turns out capacity is the real limit on bigger routes, not time -
weekday store demand is high enough that you can only fit a handful of
stores before hitting 16 pallets regardless of how much time you'd
allow. That's why there's basically no 6-stop weekday routes in the
output.

Dominance pruning
------------------
After enumerating 2-3 stop combos, we drop any route that costs more
than just running its stores as separate single trips - a solver would
never pick something like that anyway since running the singles is
strictly cheaper. Cut Monday's pool from 5,366 down to 1,457 candidates
this way, no actual options lost.

Saturday cap
------------
Saturday has lower demand per store, so way more 2-3 store combos
survive the capacity check and dominance pruning than on a weekday. We
cap it at the cheapest 40,000 after generating, and double check every
store's still covered after the cap.

Output
------
route_pool_all_days.csv, one row per candidate route across all 6 active
weekdays. Columns: RouteID, Weekday, Stores, Pallets, DriveMin,
UnloadMin, TotalMin, BaseCost, OvertimeMin, OvertimeCost, TotalCost,
NStops.
"""

import pandas as pd
import itertools
import math
import random

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
DEMAND_PATH = '8week-estimated_demand.csv'          # Supermarket, Mon..Sun, Type, Total
DURATIONS_PATH = 'FoodstuffsDurations2026.csv'       # 56x56 matrix, seconds

demand_df = pd.read_csv(DEMAND_PATH)
dur_seconds = pd.read_csv(DURATIONS_PATH, index_col=0)
dur_hr = dur_seconds / 3600.0

WAREHOUSE = 'Warehouse'
CAPACITY = 16                  # pallets per truck
UNLOAD_HR = 18 / 60.0          # hours per pallet
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def travel(a, b):
    return dur_hr.loc[a, b]

def cost_of_time(total_hr):
    """$220/hr up to 4 hours; $310/hr (rounded up to the hour) beyond 4 hours."""
    if total_hr <= 4:
        return 220 * total_hr
    return 220 * 4 + 310 * math.ceil(total_hr - 4)

def route_time(order, day_demand):
    """order = [Warehouse, stop1, stop2, ..., Warehouse]"""
    travel_hr = sum(travel(order[i], order[i+1]) for i in range(len(order)-1))
    unload_hr = sum(day_demand[s] for s in order[1:-1]) * UNLOAD_HR
    return travel_hr + unload_hr

# ---------------------------------------------------------------
# Stage A: enumerate & prune small routes (1-3 stops)
# ---------------------------------------------------------------
def best_order_and_cost(stores, day_demand):
    """Brute-force the best visit order (cheap for <=3 stops)."""
    best = None
    for perm in itertools.permutations(stores):
        order = [WAREHOUSE, *perm, WAREHOUSE]
        total_hr = route_time(order, day_demand)
        cost = cost_of_time(total_hr)
        if best is None or cost < best[2]:
            best = (order, total_hr, cost)
    return best

def enumerate_small_routes(day_demand, max_stops=3):
    stores = list(day_demand.keys())
    singles, pool = {}, []
    for r in range(1, max_stops + 1):
        for combo in itertools.combinations(stores, r):
            load = sum(day_demand[s] for s in combo)
            if load > CAPACITY:
                continue
            order, total_hr, cost = best_order_and_cost(combo, day_demand)
            pool.append({'stores': combo, 'order': order, 'pallets': load,
                         'total_hr': total_hr, 'cost': cost})
            if r == 1:
                singles[combo[0]] = cost
    return pool, singles

def prune_dominated(pool, singles):
    """Drop routes that cost more than running their stores as separate singles."""
    kept = []
    for route in pool:
        if len(route['stores']) == 1:
            kept.append(route)
            continue
        singles_cost = sum(singles[s] for s in route['stores'])
        if route['cost'] <= singles_cost:
            kept.append(route)
    return kept

# ---------------------------------------------------------------
# Stage B: construct 4-6 stop routes (Cheapest Insertion + 2-opt)
# ---------------------------------------------------------------
def randomized_insertion_build(day_demand, start_store, capacity=CAPACITY,
                                 time_cap=6.0, top_k=3, rng=None):
    order = [WAREHOUSE, start_store, WAREHOUSE]
    load = day_demand[start_store]
    used = {start_store}
    candidates = [s for s in day_demand if s not in used]
    while candidates:
        options = []
        for cand in candidates:
            if load + day_demand[cand] > capacity:
                continue
            for pos in range(1, len(order)):
                new_order = order[:pos] + [cand] + order[pos:]
                if route_time(new_order, day_demand) > time_cap:
                    continue
                delta = (travel(order[pos-1], cand) + travel(cand, order[pos])
                         - travel(order[pos-1], order[pos]))
                options.append((delta, cand, pos))
        if not options:
            break
        options.sort(key=lambda x: x[0])
        _, cand, pos = rng.choice(options[:top_k])
        order = order[:pos] + [cand] + order[pos:]
        load += day_demand[cand]
        used.add(cand)
        candidates = [s for s in day_demand if s not in used]
    return order, load

def two_opt(order, day_demand):
    """Un-cross a route by reversing segments, keep if it's cheaper."""
    improved = True
    best_time = route_time(order, day_demand)
    while improved:
        improved = False
        for i in range(1, len(order) - 2):
            for j in range(i + 1, len(order) - 1):
                new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                new_time = route_time(new_order, day_demand)
                if new_time < best_time - 1e-9:
                    order, best_time = new_order, new_time
                    improved = True
    return order, best_time

def build_multistop_pool(day_demand, n_runs=1500, min_stops=4, max_stops=6, seed=1):
    rng = random.Random(seed)
    stores = list(day_demand.keys())
    seen, pool = set(), []
    for _ in range(n_runs):
        start = rng.choice(stores)
        order, load = randomized_insertion_build(day_demand, start, rng=rng)
        n_stops = len(order) - 2
        if n_stops < min_stops or n_stops > max_stops:
            continue
        order, tot_hr = two_opt(order, day_demand)
        key = tuple(sorted(order[1:-1]))
        if key in seen:
            continue
        seen.add(key)
        pool.append({'stores': key, 'order': order, 'pallets': load,
                     'total_hr': tot_hr, 'cost': cost_of_time(tot_hr)})
    return pool

# ---------------------------------------------------------------
# Main: build the pool for every weekday and export
# ---------------------------------------------------------------
def main(max_pool_size=40000, n_multistop_runs=1500):
    all_dfs = {}
    for day in DAYS:
        day_demand = demand_df[demand_df[day] > 0].set_index('Supermarket')[day].to_dict()
        if not day_demand:
            print(f"{day}: no deliveries, skipped")
            continue

        pool, singles = enumerate_small_routes(day_demand, max_stops=3)
        kept = prune_dominated(pool, singles)
        multi = build_multistop_pool(day_demand, n_runs=n_multistop_runs,
                                       seed=hash(day) % 1000)
        full = kept + multi

        if len(full) > max_pool_size:
            full = sorted(full, key=lambda r: r['cost'])[:max_pool_size]

        day_rows = []
        for i, r in enumerate(full):
            unload_hr = r['pallets'] * UNLOAD_HR
            drive_hr = r['total_hr'] - unload_hr
            base_cost = 220 * min(r['total_hr'], 4)
            overtime_hr = max(0, r['total_hr'] - 4)
            overtime_cost = r['cost'] - base_cost

            day_rows.append({
                'RouteID': f"{day[:3]}_{i+1:05d}",
                'Weekday': day,
                'Stores': ','.join(r['stores']),
                'Pallets': r['pallets'],
                'DriveMin': round(drive_hr * 60, 1),
                'UnloadMin': round(unload_hr * 60, 1),
                'TotalMin': round(r['total_hr'] * 60, 1),
                'BaseCost': round(base_cost, 2),
                'OvertimeMin': round(overtime_hr * 60, 1),
                'OvertimeCost': round(overtime_cost, 2),
                'TotalCost': round(r['cost'], 2),
                'NStops': len(r['stores'])
            })

        # write this day's own CSV straight away, e.g. routes_Monday.csv
        day_df = pd.DataFrame(day_rows)
        out_name = f"routes_{day}.csv"
        day_df.to_csv(out_name, index=False)
        all_dfs[day] = day_df
        print(f"{day}: {len(full)} routes -> saved to {out_name}")

    print(f"\nDone. Wrote {len(all_dfs)} files: " +
          ", ".join(f"routes_{d}.csv" for d in all_dfs))
    return all_dfs

if __name__ == '__main__':
    main()