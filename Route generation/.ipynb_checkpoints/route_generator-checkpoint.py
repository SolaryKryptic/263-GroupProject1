"""
Route generator for the Foodstuffs trucking problem -- Monday through
Saturday (Sunday has zero demand across the network).

Produces a pool of candidate routes for a chosen day, using randomized-
greedy construction: distance-weighted stop selection, guaranteed forced-
start coverage per store, and probabilistic early-stopping. Calibrated
defaults (ALPHA=3.0, EARLY_STOP_PROB=0.12) were tuned empirically against
Saturday (the day with a genuinely large search space) using multi-seed
averaging -- see CONFIG section to override any of them.

Feed the output CSV into solve_lp.py to actually pick the best combination
of routes.

Usage
-----
    python3 generate_routes.py Monday
    python3 generate_routes.py Wednesday --budget 5000 --alpha 2.5

Or import and call generate_for_day() directly from other code.
"""

import argparse
import csv
import random
from pathlib import Path

# ----------------------------- CONFIG ---------------------------------- #

DURATIONS_CSV = "FoodstuffsDurations2026.csv"
DEMAND_CSV = "0_5ayush6week-estimated_demand.csv"
OUTPUT_DIR = "engsci263"

WAREHOUSE = "Warehouse"
CAPACITY_PALLETS = 16
UNLOADING_SEC_PER_PALLET = 18 * 60   # 18 minutes per pallet, charged at each stop

# Calibrated defaults (see module docstring)
ALPHA = 3.0                  # distance-weighting strength: 0=random, higher=greedier
EARLY_STOP_PROB = 0.12       # chance per step of ending a route early, capacity room or not
FORCED_STARTS_PER_STORE = 10 # guaranteed independent route-builds per store, as stop #1
TOTAL_BUDGET = 3000          # total route-build attempts (forced + unforced together)

VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# ------------------------------------------------------------------------ #


def load_durations(path):
    """durations[a][b] = travel time in seconds, for every location pair."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        locations = header[1:]
        durations = {loc: {} for loc in locations}
        for row in reader:
            origin = row[0]
            for loc, val in zip(locations, row[1:]):
                durations[origin][loc] = float(val)
    return durations


def load_demand(path, day):
    """demand[store] = pallets for the given day, only stores with demand > 0."""
    demand = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qty = int(row[day])
            if qty > 0:
                demand[row["Supermarket"]] = qty
    return demand


def weighted_pick(candidates, rng, alpha):
    """candidates: list of (travel_time_sec, store). Picks one, probability
    proportional to 1/travel_time^alpha -- closer stores more likely, but
    every feasible candidate has a nonzero chance."""
    weights = [1.0 / (t ** alpha) if t > 0 else 1.0 for t, _ in candidates]
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for w, (_, store) in zip(weights, candidates):
        upto += w
        if upto >= r:
            return store
    return candidates[-1][1]


def build_one_route(durations, demand, all_stores, rng, alpha,
                     forced_start=None, early_stop_prob=0.0):
    """
    Builds one route: Warehouse -> ... -> Warehouse. Draws candidates from
    the FULL store list every step (independent of any other route built
    elsewhere). The only hard feasibility check is the 16-pallet cap --
    duration is uncapped, tracked only for downstream cost calculation.
    """
    current = WAREHOUSE
    time_elapsed = 0.0
    load = 0
    stops = []
    visited = set()

    def unload_time(store):
        return demand[store] * UNLOADING_SEC_PER_PALLET

    if forced_start is not None and demand[forced_start] <= CAPACITY_PALLETS:
        stops.append(forced_start)
        visited.add(forced_start)
        time_elapsed += durations[current][forced_start] + unload_time(forced_start)
        load += demand[forced_start]
        current = forced_start

    while True:
        if stops and early_stop_prob > 0.0 and rng.random() < early_stop_prob:
            break
        candidates = [(durations[current][s], s) for s in all_stores
                      if s not in visited and load + demand[s] <= CAPACITY_PALLETS]
        if not candidates:
            break
        chosen = weighted_pick(candidates, rng, alpha)
        time_elapsed += durations[current][chosen] + unload_time(chosen)
        load += demand[chosen]
        stops.append(chosen)
        visited.add(chosen)
        current = chosen

    time_elapsed += durations[current][WAREHOUSE]
    if not stops:
        return None
    return {
        "stops": tuple(stops),
        "num_stops": len(stops),
        "total_pallets": load,
        "total_duration_sec": round(time_elapsed, 2),
    }


def generate_for_day(day, durations=None, demand=None, rng=None,
                      alpha=ALPHA, early_stop_prob=EARLY_STOP_PROB,
                      forced_per_store=FORCED_STARTS_PER_STORE,
                      total_budget=TOTAL_BUDGET):
    """
    Generate the candidate route pool for one day. Returns a list of route
    dicts. `total_budget` covers BOTH forced-start and unforced attempts --
    forced attempts = forced_per_store * num_stores, the remainder goes to
    unforced draws.
    """
    if day not in VALID_DAYS:
        raise ValueError(f"'{day}' has no demand or isn't a valid day. Choose from {VALID_DAYS}.")

    if durations is None:
        durations = load_durations(DURATIONS_CSV)
    if demand is None:
        demand = load_demand(DEMAND_CSV, day)
    if rng is None:
        rng = random.Random()  # unseeded -- genuinely random each call

    stores = list(demand.keys())
    n_forced_total = forced_per_store * len(stores)
    n_unforced = max(0, total_budget - n_forced_total)

    pool = {}
    for store in stores:
        for _ in range(forced_per_store):
            r = build_one_route(durations, demand, stores, rng, alpha,
                                 forced_start=store, early_stop_prob=early_stop_prob)
            if r:
                pool[r["stops"]] = r
    for _ in range(n_unforced):
        r = build_one_route(durations, demand, stores, rng, alpha,
                             forced_start=None, early_stop_prob=early_stop_prob)
        if r:
            pool[r["stops"]] = r

    return list(pool.values())


def write_routes_csv(routes, day, output_dir=OUTPUT_DIR):
    path = Path(output_dir) / f"routes_{day}.csv"
    routes = sorted(routes, key=lambda r: r["total_duration_sec"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["route_id", "stops", "num_stops", "total_pallets",
                          "total_duration_sec", "total_duration_hms"])
        for i, r in enumerate(routes, start=1):
            sec = r["total_duration_sec"]
            hrs, mins, secs = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
            writer.writerow([f"{day[:3]}_{i:05d}", ";".join(r["stops"]), r["num_stops"],
                              r["total_pallets"], sec, f"{hrs:02d}:{mins:02d}:{secs:02d}"])
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate candidate routes for a given day.")
    parser.add_argument("day", choices=VALID_DAYS)
    parser.add_argument("--budget", type=int, default=TOTAL_BUDGET,
                         help="total route-build attempts (forced + unforced)")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                         help="distance-weighting strength")
    parser.add_argument("--early-stop", type=float, default=EARLY_STOP_PROB,
                         help="probability of ending a route early each step")
    parser.add_argument("--forced-per-store", type=int, default=FORCED_STARTS_PER_STORE,
                         help="guaranteed route-builds per store as the forced first stop")
    parser.add_argument("--seed", type=int, default=None,
                         help="fix the random seed for reproducibility (default: random)")
    args = parser.parse_args()

    durations = load_durations(DURATIONS_CSV)
    demand = load_demand(DEMAND_CSV, args.day)
    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    routes = generate_for_day(args.day, durations, demand, rng,
                               alpha=args.alpha, early_stop_prob=args.early_stop,
                               forced_per_store=args.forced_per_store,
                               total_budget=args.budget)
    path = write_routes_csv(routes, args.day)

    total_pallets = sum(demand.values())
    print(f"{args.day}: {len(demand)} stores, {total_pallets} pallets, "
          f"budget={args.budget} (alpha={args.alpha}, early_stop={args.early_stop}, "
          f"forced/store={args.forced_per_store}) -> {len(routes)} unique routes")
    print(f"Written to {path}")


if __name__ == "__main__":
    main()


    
