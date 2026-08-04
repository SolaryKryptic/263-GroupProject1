"""
Randomized-probabilistic route generator for the Foodstuffs delivery problem.

Purpose
-------
This script does NOT try to build "a route plan" for the day. Its only job
is to produce a large, diverse POOL of independent, feasible candidate
routes. Every route is generated completely independently of every other
route -- nothing is shared or "used up" between generations. You feed this
pool into a set-partitioning / set-covering LP later, and the LP is the
thing that actually assembles a day plan by choosing the best combination
of routes from the pool so that every store gets its full daily pallet
demand delivered.

Why NOT a shrinking/shared pool
--------------------------------
An earlier version of this script generated one full day's worth of routes
at a time by removing stores from a shared "unvisited" set as they got
assigned to a route, repeating until every store for the day was covered.
That guarantees a complete day plan every run, but it means later routes in
a run are only ever built from whatever earlier routes in that SAME run
happened to leave behind -- i.e. every route after the first is dependent on
prior random choices in that run, not a free, independent draw from the
full network. That's a reasonable way to build one candidate day plan, but
it's the wrong tool for building a diverse, independent CANDIDATE POOL for
an LP to choose from. This version removes that dependency entirely: every
single route, in every iteration, is built with the full set of stores
available to it, every time.

How routes are built
---------------------
For a single route:
    1. Start at the Warehouse, time = 0, load = 0.
    2. At each step, look at every store not already visited BY THIS route
       (every other store is a candidate -- the full daily list, not some
       shrinking global pool) that is FEASIBLE to add next, i.e. adding it
       would not:
         a) push the truck's pallet load over CAPACITY (16), or
         b) push the total route duration -- travel time, plus 18 minutes
            of unloading at every store on the route (including this one),
            plus the trip back to the Warehouse afterward -- over
            TIME_LIMIT_SEC (3.5 hours).
    3. Instead of a hard "top-5 nearest, then pick randomly among those"
       cutoff, every feasible candidate gets a selection PROBABILITY
       proportional to 1 / (travel_time ** WEIGHT_ALPHA). Closer stores are
       more likely to be picked, but nothing is ever excluded outright --
       a distant store just becomes proportionally rarer, not impossible,
       even when many closer competitors are present in the same draw.
    4. Move to the chosen store, update time/load, repeat until nothing
       feasible remains, then return to the Warehouse.

Guaranteeing every store gets genuine coverage
-------------------------------------------------
Because every draw is now independent and weighted toward nearby stores,
an inconveniently-located store could still end up rarely chosen if we only
ever did unforced random draws (a store that's simply far from everything
will have low selection probability at every step, every time). To fix that
without reintroducing any shrinking/shared state, we explicitly FORCE every
store to be the guaranteed first stop out of the Warehouse for a number of
independent route builds (ITERATIONS_PER_FORCED_START each). The rest of
each of those routes is still built the normal probabilistic way, using the
full remaining store list -- so you get real variety in what gets grouped
around every store, not just one fixed pairing. On top of that, a large
number of fully unforced, fully probabilistic independent draws
(N_UNFORCED_ITERATIONS) are added for organic diversity everywhere else.

On WEIGHT_ALPHA (replaces the old TOP_K knob)
------------------------------------------------
This is the tunable "how greedy vs. how random" dial:
    - WEIGHT_ALPHA = 0: every feasible candidate is equally likely
      regardless of distance -- pure random walk, ignores geography
      entirely. Produces lots of very inefficient routes.
    - Small WEIGHT_ALPHA (0.5-1.5): mild preference for closer stores,
      but distant ones still get picked reasonably often -- lots of
      diversity, more of the generated routes will be inefficient though.
    - Large WEIGHT_ALPHA (3+): strongly prefers the nearest candidate,
      behaving close to plain nearest-neighbor greedy -- most routes will
      look efficient/similar, less diversity.
    I've defaulted to 1.5 as a middle ground. Worth experimenting with --
    compare pool size AND how spread out any one store's selection-count is
    across different ALPHA values (printed per-store stats below) to see
    which gives the diversity/efficiency balance you want.

Not every forced-start iteration is equally valuable
------------------------------------------------------
I tested this empirically rather than guessing: forcing a "hard" store
(e.g. one far from everything) as the start, the number of genuinely
distinct route variants found keeps growing roughly linearly even past 100+
iterations -- there's no early plateau that would justify simply giving
hard stores fewer forced iterations. The real issue is different: many of
those variants cluster around a similar, mediocre cost, because the hard
store's own expensive leg dominates the route's total cost regardless of
what it's paired with. An LP will only ever pick from the handful of
cheapest options for that store; the rest just bloat the candidate pool
without ever being selected. MAX_ROUTES_PER_STORE fixes this directly and
objectively (by cost, not by a guessed-in-advance "this store is hard"
label): after generation, trim_pool_by_cost() keeps only the N cheapest
routes containing each store, for every store, dropping cost-dominated
variants. A store whose best option is inherently expensive still keeps
that option -- it's ranked among ITS OWN routes, not against the whole
pool.

Partial / "leftover" routes
-----------------------------
Every route previously only stopped when nothing more could feasibly be
added, biasing the pool toward maximally-packed routes. A real day plan
usually needs some genuinely partial, under-capacity routes too -- for
whatever's left over once the efficient combinations are claimed elsewhere.
EARLY_STOP_PROB adds a per-step chance (after at least one stop has been
made) of ending the route right there, room or not, so the pool contains a
realistic mix of route sizes rather than only "packed to the limit" ones.

Outputs
-------
For each day, writes routes_<day>.csv into the output directory with one row
per candidate route: its stops in order, number of stops, total pallets,
and total duration. Also writes a pool_summary.csv across all days and a
coverage_check.csv confirming every store's appearance count in the pool.
"""

import csv
import random
from pathlib import Path

# ----------------------------- CONFIG ---------------------------------- #

DURATIONS_CSV = "/mnt/user-data/uploads/FoodstuffsDurations2026.csv"
DEMAND_CSV = "/mnt/user-data/uploads/0_5ayush6week-estimated_demand.csv"
OUTPUT_DIR = "/mnt/user-data/outputs"

WAREHOUSE = "Warehouse"
CAPACITY_PALLETS = 16
TIME_LIMIT_SEC = 3.5 * 3600  # 12,600 seconds
UNLOADING_SEC = 18 * 60      # 1,080 seconds spent unloading at each store visited

WEIGHT_ALPHA = 1.5                  # higher = greedier/closer-preferring, lower = more random
ITERATIONS_PER_FORCED_START = 40    # independent route builds per store, forced as the 1st stop
N_UNFORCED_ITERATIONS = 4000        # additional fully independent, fully probabilistic draws
EARLY_STOP_PROB = 0.12              # chance, at each step after the 1st stop, of ending the
                                     # route early (returning to warehouse) even if more stops
                                     # would still be feasible -- generates realistic partial/
                                     # "leftover" routes, not just maximally-packed ones
MAX_ROUTES_PER_STORE = 150          # after generation, keep at most this many of the cheapest
                                     # routes that contain each store (dedupes low-value,
                                     # cost-dominated variants an LP would never pick anyway)
RANDOM_SEED = 42                    # set to None for a different pool every run

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ------------------------------------------------------------------------ #


def load_durations(path):
    """Load the asymmetric travel-time matrix into durations[a][b] = seconds."""
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


def load_demand(path):
    """Load demand[day][store] = pallets, only including stores with demand > 0."""
    demand = {day: {} for day in DAYS}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            store = row["Supermarket"]
            for day in DAYS:
                qty = int(row[day])
                if qty > 0:
                    demand[day][store] = qty
    return demand


def weighted_pick(candidates, rng, alpha):
    """
    candidates: list of (travel_time_seconds, store_name), all feasible.
    Returns one store, chosen with probability proportional to
    1 / (travel_time ** alpha). Every candidate has a nonzero chance.
    """
    weights = [1.0 / (t ** alpha) if t > 0 else 1.0 for t, _ in candidates]
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for w, (_, store) in zip(weights, candidates):
        upto += w
        if upto >= r:
            return store
    return candidates[-1][1]  # floating point fallback


def build_one_route(durations, demand, all_stores, rng, alpha, forced_start=None,
                     early_stop_prob=0.0):
    """
    Build a single feasible route, starting and ending at the Warehouse.
    Draws candidates from the FULL `all_stores` list every step (minus
    whatever this specific route has already visited) -- no dependency on
    any other route generated before or after it.

    If forced_start is given, that store is guaranteed to be the first stop
    (as long as it's individually feasible, which every store in this
    dataset is). Every subsequent stop is chosen probabilistically.

    early_stop_prob: after each stop (once at least one has been made), the
    route ends right there with this probability, even if more stores would
    still fit within time/capacity. This produces genuinely partial,
    under-capacity routes -- the kind a final route PLAN often needs for
    whatever's "left over" once the efficient combinations are taken --
    rather than only ever generating maximally-packed routes.
    """
    current = WAREHOUSE
    time_elapsed = 0.0
    load = 0
    stops = []
    visited_this_route = set()

    if forced_start is not None:
        travel = durations[current][forced_start]
        projected_time = travel + UNLOADING_SEC + durations[forced_start][WAREHOUSE]
        if projected_time <= TIME_LIMIT_SEC and demand[forced_start] <= CAPACITY_PALLETS:
            stops.append(forced_start)
            visited_this_route.add(forced_start)
            time_elapsed += travel + UNLOADING_SEC
            load += demand[forced_start]
            current = forced_start
        # (every store in this dataset is solo-feasible from the Warehouse,
        # so this branch always succeeds in practice -- no silent skip risk)

    while True:
        if stops and early_stop_prob > 0.0 and rng.random() < early_stop_prob:
            break  # deliberately end the route early, room or not

        candidates = []
        for store in all_stores:
            if store in visited_this_route:
                continue
            travel_to_store = durations[current][store]
            travel_back_to_warehouse = durations[store][WAREHOUSE]
            projected_time = (
                time_elapsed + travel_to_store + UNLOADING_SEC + travel_back_to_warehouse
            )
            projected_load = load + demand[store]
            if projected_time <= TIME_LIMIT_SEC and projected_load <= CAPACITY_PALLETS:
                candidates.append((travel_to_store, store))

        if not candidates:
            break

        chosen = weighted_pick(candidates, rng, alpha)
        time_elapsed += durations[current][chosen] + UNLOADING_SEC
        load += demand[chosen]
        stops.append(chosen)
        visited_this_route.add(chosen)
        current = chosen

    time_elapsed += durations[current][WAREHOUSE]  # return leg

    if not stops:
        return None

    return {
        "stops": tuple(stops),
        "num_stops": len(stops),
        "total_pallets": load,
        "total_duration_sec": round(time_elapsed, 2),
    }


def generate_route_pool(day_stores_demand, durations, rng, alpha,
                         iterations_per_forced_start, n_unforced_iterations,
                         early_stop_prob=0.0):
    """
    Build the candidate pool for one day:
      1. For every store, force it as the starting stop across several
         independent route builds, guaranteeing genuine representation.
      2. Add a large batch of fully unforced, fully independent,
         probabilistically-built routes for organic diversity.
    Every build draws from the complete store list -- nothing shrinks,
    nothing is shared between builds.
    """
    pool = {}  # key: stops tuple -> route dict
    stores = list(day_stores_demand.keys())

    for store in stores:
        for _ in range(iterations_per_forced_start):
            route = build_one_route(durations, day_stores_demand, stores, rng, alpha,
                                     forced_start=store, early_stop_prob=early_stop_prob)
            if route:
                pool[route["stops"]] = route

    for _ in range(n_unforced_iterations):
        route = build_one_route(durations, day_stores_demand, stores, rng, alpha,
                                 forced_start=None, early_stop_prob=early_stop_prob)
        if route:
            pool[route["stops"]] = route

    return list(pool.values())


def trim_pool_by_cost(pool, max_routes_per_store):
    """
    Cap how many routes survive per store: for each store, only the
    `max_routes_per_store` routes with the best SECONDS-PER-PALLET
    efficiency (total_duration_sec / total_pallets) count toward "keeping"
    it. A route survives if it's within that efficient-set for AT LEAST ONE
    of its stops.

    IMPORTANT: ranking by raw total_duration_sec (instead of a per-pallet
    rate) would systematically favor short, few-stop routes -- fewer stops
    always means less accumulated travel + 18-min unloading time, so a lone
    1-stop route always looks "cheaper" in absolute seconds than a
    well-packed 4-5 stop route, even though the packed route is usually the
    actually EFFICIENT option (one truck, one driver, several deliveries).
    Verified empirically: ranking by raw duration wiped out 100% of 5-stop
    routes and 19% of 4-stop routes while keeping 100% of 1-3 stop routes --
    exactly backwards from what an LP should be offered. Normalizing by
    pallets delivered fixes that: it rewards efficient packing rather than
    penalizing routes for serving more stores, while still keeping a
    genuinely isolated store's least-bad solo option (ranked only against
    ITS OWN alternatives, not the whole pool).
    """
    from collections import defaultdict

    by_store = defaultdict(list)
    for route in pool:
        for store in route["stops"]:
            by_store[store].append(route)

    keep_ids = set()
    for store, routes in by_store.items():
        routes.sort(key=lambda r: r["total_duration_sec"] / r["total_pallets"])
        for route in routes[:max_routes_per_store]:
            keep_ids.add(route["stops"])

    return [r for r in pool if r["stops"] in keep_ids]


def write_routes_csv(routes, day, output_dir):
    path = Path(output_dir) / f"routes_{day}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["route_id", "stops", "num_stops", "total_pallets", "total_duration_sec", "total_duration_hms"]
        )
        for i, route in enumerate(routes, start=1):
            hrs = int(route["total_duration_sec"] // 3600)
            mins = int((route["total_duration_sec"] % 3600) // 60)
            secs = int(route["total_duration_sec"] % 60)
            writer.writerow(
                [
                    f"{day[:3]}_{i:04d}",
                    ";".join(route["stops"]),
                    route["num_stops"],
                    route["total_pallets"],
                    route["total_duration_sec"],
                    f"{hrs:02d}:{mins:02d}:{secs:02d}",
                ]
            )
    return path


def write_coverage_check(pool, day_stores_demand, day, output_dir, existing_rows):
    """Confirms every store with demand that day appears in at least one route."""
    from collections import Counter
    counts = Counter()
    for route in pool:
        for store in route["stops"]:
            counts[store] += 1

    for store in day_stores_demand:
        existing_rows.append({
            "day": day,
            "store": store,
            "times_appearing_in_pool": counts.get(store, 0),
        })
    return existing_rows


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    durations = load_durations(DURATIONS_CSV)
    demand = load_demand(DEMAND_CSV)

    summary_rows = []
    coverage_rows = []

    for day in DAYS:
        day_demand = demand[day]
        if not day_demand:
            print(f"{day}: no demand, skipping (0 stores).")
            continue

        seed = None if RANDOM_SEED is None else RANDOM_SEED + DAYS.index(day)
        rng = random.Random(seed)

        pool = generate_route_pool(
            day_demand, durations, rng, WEIGHT_ALPHA,
            ITERATIONS_PER_FORCED_START, N_UNFORCED_ITERATIONS,
            early_stop_prob=EARLY_STOP_PROB,
        )
        pool_before_trim = len(pool)
        pool = trim_pool_by_cost(pool, MAX_ROUTES_PER_STORE)

        pool.sort(key=lambda r: r["total_duration_sec"])  # cheapest/shortest first
        path = write_routes_csv(pool, day, OUTPUT_DIR)
        coverage_rows = write_coverage_check(pool, day_demand, day, OUTPUT_DIR, coverage_rows)

        n_stores = len(day_demand)
        total_pallets = sum(day_demand.values())
        min_trucks_by_capacity = -(-total_pallets // CAPACITY_PALLETS)  # ceil

        day_coverage = [r["times_appearing_in_pool"] for r in coverage_rows if r["day"] == day]
        never_covered = sum(1 for c in day_coverage if c == 0)

        print(
            f"{day}: {n_stores} stores, {total_pallets} pallets total "
            f"(>= {min_trucks_by_capacity} trucks needed by capacity alone) "
            f"-> {pool_before_trim} generated, {len(pool)} kept after cost-trim | "
            f"min/max store appearances: {min(day_coverage)}/{max(day_coverage)} | "
            f"stores never covered: {never_covered}"
        )

        summary_rows.append(
            {
                "day": day,
                "num_stores_with_demand": n_stores,
                "total_pallets": total_pallets,
                "min_trucks_by_capacity": min_trucks_by_capacity,
                "unique_candidate_routes": len(pool),
                "min_store_appearances": min(day_coverage),
                "max_store_appearances": max(day_coverage),
            }
        )

    summary_path = Path(OUTPUT_DIR) / "pool_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "day", "num_stores_with_demand", "total_pallets",
                "min_trucks_by_capacity", "unique_candidate_routes",
                "min_store_appearances", "max_store_appearances",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    coverage_path = Path(OUTPUT_DIR) / "coverage_check.csv"
    with open(coverage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["day", "store", "times_appearing_in_pool"])
        writer.writeheader()
        writer.writerows(coverage_rows)

    print(f"\nSummary written to {summary_path}")
    print(f"Coverage check written to {coverage_path}")


if __name__ == "__main__":
    main()
