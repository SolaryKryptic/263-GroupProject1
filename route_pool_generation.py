"""
route_pool_generation.py

Generates a LARGE POOL of feasible candidate truck trips (routes) for each
weekday (Mon-Sat) using the chosen demand estimates
(estimations/0.5ayush6week-estimated_demand.csv) and the travel-time matrix
(FoodstuffsDurations2026.csv).

A trip is a sequence of stores starting and ending at the Mt Roskill
distribution Warehouse. Trips are generated WITHOUT choosing which set of
trips is actually driven - selecting the cheapest covering subset (set-cover /
MILP) is a separate later step.

Modelling rules (from the problem statement):
  - truck capacity: 16 pallets
  - a store's daily demand cannot be split across trucks (one stop per trip)
  - unload time: 18 min per pallet
  - trip duration = driving time (duration matrix, seconds) + unload time
  - scheduled cost: $220 / hour of trip time
  - soft 3.5h (210 min) cap; any time beyond 3.5h is overtime at $310 / hour
  - fleet: 20 trucks x 2 shifts = 40 trip slots per day (checked later, not here)

Generation methods (all produce feasible trips):
  1. single-stop trips (every store with demand > 0)  - guarantees coverage
  2. nearest-neighbour insertions per seed store
  3. randomised Clarke-Wright savings restarts
  4. random spatially-clustered subsets (nearest-K stores)

All trips are de-duplicated on the canonical store set, keeping the cheapest
ordering found. Trips are written to routes/pool_<Weekday>.csv
"""

import os

import numpy as np
import pandas as pd

DEMAND_FILE = "estimations/0.5ayush6week-estimated_demand.csv"
DURATIONS_FILE = "FoodstuffsDurations2026.csv"
OUTPUT_DIR = "routes"
WAREHOUSE = "Warehouse"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

TRUCK_CAPACITY = 16
UNLOAD_MIN_PER_PALLET = 18
BASE_TRIP_CAP_MIN = 210  # 3.5h
BASE_COST_PER_HOUR = 220.0
OVERTIME_COST_PER_HOUR = 310.0

# Multi-stop trips are capped at this duration (single-stop trips are exempt
# because large Pak 'n Save stores legitimately exceed the soft cap).
MAX_TRIP_MIN = 360  # 6h - generous cap so the pool contains many candidates
MAX_STOPS = 6

SAVINGS_RESTARTS = 150
RANDOM_SUBSETS = 6000
RANDOM_K = 10  # candidate neighbours when building random subsets
MAX_POOL_PER_DAY = 40000


def build_cost(drive_sec, pallets):
    """Returns (total_min, base_cost, overtime_min, overtime_cost, total_cost)."""
    unload_min = UNLOAD_MIN_PER_PALLET * pallets
    total_min = drive_sec / 60.0 + unload_min
    base_min = min(total_min, BASE_TRIP_CAP_MIN)
    overtime_min = max(0.0, total_min - BASE_TRIP_CAP_MIN)
    base_cost = base_min / 60.0 * BASE_COST_PER_HOUR
    overtime_cost = overtime_min / 60.0 * OVERTIME_COST_PER_HOUR
    return total_min, base_cost, overtime_min, overtime_cost, base_cost + overtime_cost


def route_time(order, d_sec, wh_idx):
    """Drive seconds for a route visiting `order` (list of node idxs) then returning."""
    prev = wh_idx
    total = 0
    for node in order:
        total += d_sec[prev, node]
        prev = node
    total += d_sec[prev, wh_idx]
    return total


def nearest_neighbour_order(nodes, d_sec, wh_idx):
    """Order a set of nodes by always driving to the nearest unvisited store."""
    order = [wh_idx]
    remaining = list(nodes)
    while remaining:
        cur = order[-1]
        nxt = min(remaining, key=lambda n: d_sec[cur, n])
        order.append(nxt)
        remaining.remove(nxt)
    return order[1:]


def is_feasible(order, pallets, d_sec, wh_idx, single_ok=True):
    total = sum(pallets[n] for n in order)
    if total > TRUCK_CAPACITY:
        return False
    drive_sec = route_time(order, d_sec, wh_idx)
    if len(order) == 1 and single_ok:
        return True
    if drive_sec / 60.0 + UNLOAD_MIN_PER_PALLET * total > MAX_TRIP_MIN:
        return False
    return True


def add_trip(pool, stores, order, pallets, d_sec, wh_idx, name_of, weekday):
    """Insert a trip keyed on its canonical store set, keeping the cheapest one."""
    key = tuple(sorted(stores))
    drive_sec = route_time(order, d_sec, wh_idx)
    total_p = sum(pallets[n] for n in stores)
    total_min, base, ot_min, ot_cost, total_cost = build_cost(drive_sec, total_p)
    existing = pool.get(key)
    if existing is None or total_cost < existing["TotalCost"]:
        pool[key] = {
            "Weekday": weekday,
            "Stores": ",".join(name_of[n] for n in order),
            "Pallets": total_p,
            "DriveMin": round(drive_sec / 60.0, 1),
            "UnloadMin": round(UNLOAD_MIN_PER_PALLET * total_p, 1),
            "TotalMin": round(total_min, 1),
            "BaseCost": round(base, 2),
            "OvertimeMin": round(ot_min, 1),
            "OvertimeCost": round(ot_cost, 2),
            "TotalCost": round(total_cost, 2),
            "NStops": len(order),
        }


def savings_cw_restart(active, pallets, d_sec, wh_idx, name_of, pool, rng, weekday):
    """One randomised Clarke-Wright pass. Collects every intermediate feasible route."""
    n = len(name_of)
    wh = wh_idx
    is_active = np.zeros(n, dtype=bool)
    is_active[active] = True
    savings = np.zeros((n, n))
    for i in range(n):
        if not is_active[i]:
            continue
        for j in range(i + 1, n):
            if not is_active[j]:
                continue
            raw = d_sec[wh, i] + d_sec[wh, j] - d_sec[i, j]
            savings[i, j] = savings[j, i] = max(raw, 0.0) * rng.uniform(0.6, 1.4)
    order = np.argsort(-savings, axis=None)
    ii, jj = np.unravel_index(order, savings.shape)

    routes = {i: [i] for i in active}
    used = set(active)

    for i, j in zip(ii, jj):
        if i == j or i in used or j in used:
            continue
        if not is_active[i] or not is_active[j]:
            continue
        ra, rb = routes[i], routes[j]
        if ra is rb:
            continue
        if not (ra[0] == i or ra[-1] == i) or not (rb[0] == j or rb[-1] == j):
            continue
        best = None
        for fa in (ra, list(reversed(ra))):
            if not (fa[0] == i or fa[-1] == i):
                continue
            for fb in (rb, list(reversed(rb))):
                if not (fb[0] == j or fb[-1] == j):
                    continue
                for merged in ([fa + fb], [fb + fa]):
                    m = merged[0]
                    if sum(pallets[x] for x in m) > TRUCK_CAPACITY:
                        continue
                    drive_sec = route_time(m, d_sec, wh)
                    tmin = drive_sec / 60.0 + UNLOAD_MIN_PER_PALLET * sum(pallets[x] for x in m)
                    if tmin > MAX_TRIP_MIN:
                        continue
                    cost = build_cost(drive_sec, sum(pallets[x] for x in m))[0]
                    if best is None or cost < best[0]:
                        best = (cost, m)
        if best is None:
            continue
        merged = best[1]
        add_trip(pool, merged, merged, pallets, d_sec, wh, name_of, weekday)
        for x in merged:
            routes[x] = merged
            used.add(x)
        used.discard(i)
        used.discard(j)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    est = pd.read_csv(DEMAND_FILE)
    est = est.set_index("Supermarket")

    dur = pd.read_csv(DURATIONS_FILE, index_col=0)
    dur = dur.loc[list(est.index) + [WAREHOUSE], list(est.index) + [WAREHOUSE]]

    store_names = list(est.index)
    idx_map = {name: i for i, name in enumerate(store_names)}
    wh_idx = len(store_names)
    idx_map[WAREHOUSE] = wh_idx
    name_of = {i: name for name, i in idx_map.items()}
    d_sec = dur.to_numpy(dtype=float)

    rng = np.random.default_rng(263)

    for weekday in WEEKDAYS:
        pallets = est[weekday].to_numpy(dtype=int)
        active = [i for i, p in enumerate(pallets) if p > 0]
        pool = {}

        for i in active:
            order = [i]
            add_trip(pool, order, order, pallets, d_sec, wh_idx, name_of, weekday)

        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                nodes = [active[a], active[b]]
                if not is_feasible(nodes, pallets, d_sec, wh_idx):
                    continue
                order = nearest_neighbour_order(nodes, d_sec, wh_idx)
                add_trip(pool, nodes, order, pallets, d_sec, wh_idx, name_of, weekday)

        dist_from_wh = d_sec[wh_idx]
        for seed in active:
            others = sorted(active, key=lambda j: d_sec[seed, j])
            for k in range(2, min(MAX_STOPS, len(active)) + 1):
                nodes = [seed] + [j for j in others if j != seed][: k - 1]
                if not is_feasible(nodes, pallets, d_sec, wh_idx):
                    continue
                order = nearest_neighbour_order(nodes, d_sec, wh_idx)
                add_trip(pool, nodes, order, pallets, d_sec, wh_idx, name_of, weekday)

        for restart in range(SAVINGS_RESTARTS):
            savings_cw_restart(active, pallets, d_sec, wh_idx, name_of, pool, rng, weekday)

        for _ in range(RANDOM_SUBSETS):
            seed = int(rng.choice(active))
            k = int(rng.integers(2, min(MAX_STOPS, len(active)) + 1))
            candidates = sorted(active, key=lambda j: d_sec[seed, j])[:RANDOM_K]
            nodes = [seed] + [int(x) for x in rng.choice([c for c in candidates if c != seed],
                                                         size=k - 1, replace=False)]
            if len(set(nodes)) < k:
                continue
            if not is_feasible(nodes, pallets, d_sec, wh_idx):
                continue
            order = nearest_neighbour_order(nodes, d_sec, wh_idx)
            add_trip(pool, nodes, order, pallets, d_sec, wh_idx, name_of, weekday)

        rows = sorted(pool.values(), key=lambda r: r["TotalCost"])[:MAX_POOL_PER_DAY]

        covered = set()
        for r in rows:
            covered |= set(r["Stores"].split(","))
        uncovered = [store_names[i] for i in active if store_names[i] not in covered]

        df = pd.DataFrame(rows)
        df.insert(0, "RouteID", [f"{weekday[:3]}_{i:05d}" for i in range(1, len(df) + 1)])
        df.to_csv(os.path.join(OUTPUT_DIR, f"pool_{weekday}.csv"), index=False)

        hist = df["NStops"].value_counts().sort_index()
        ot_count = (df["OvertimeMin"] > 0).sum()
        print(f"{weekday}: {len(df)} unique trips | coverage {100 * (len(active) - len(uncovered)) / len(active):.0f}% "
              f"({len(active)} stores)" + (f" | UNCOVERED: {uncovered}" if uncovered else ""))
        print(f"   stop histogram: " + ", ".join(f"{k}stop={int(v)}" for k, v in hist.items()))
        print(f"   overtime trips: {ot_count} | min trip cost ${df['TotalCost'].min():.2f}, "
              f"max ${df['TotalCost'].max():.2f}")

    print("\nDone. Pools written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
