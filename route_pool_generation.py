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
  v1 / v2 (heuristic):
    1. single-stop trips (every store with demand > 0)  - guarantees coverage
    2. exhaustive 2-stop pairs
    3. nearest-neighbour insertions per seed store
    4. randomised Clarke-Wright savings restarts
    5. random spatially-clustered subsets (nearest-K stores)
  v3 (exhaustive):
    - single-stop trips, plus EVERY feasible 2/3/4-stop trip (capacity + time
      checked on the best ordering found by 2-opt), so the pool is complete up
      to 4 stops - the MILP can then find a provably optimal solution
    - randomised savings + random subsets still add 5-6 stop trips
    - routes that cost more than serving their stores separately are pruned
      (they can never be in an optimal partition)

All trips are de-duplicated on the canonical store set, keeping the cheapest
ordering found.

Three pool versions are produced so they can be compared:
  - v1 (routes/pool_<Weekday>.csv): heuristic store-sets, generator ordering
  - v2 (routes_2opt/pool_<Weekday>.csv): same store-sets as v1, but each trip's
    order is tightened with 2-opt + relocate (same seed, ordering is the only
    difference)
  - v3 (v3/pool_<Weekday>.csv): exhaustive store-sets up to 4 stops, 2-opt
    ordering, dominated routes pruned
"""

import os

import numpy as np
import pandas as pd

DEMAND_FILE = "estimations/0.5ayush6week-estimated_demand.csv"
DURATIONS_FILE = "FoodstuffsDurations2026.csv"
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

USE_2OPT = False  # set per pass: tighten trip ordering with 2-opt + relocate


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


def improve_order(order, d_sec, wh_idx):
    """Tighten a trip's store ordering with 2-opt (segment reversals) and
    relocate (Or-opt) moves until no single move shortens the drive time."""
    order = list(order)
    n = len(order)
    if n < 2:
        return order
    changed = True
    while changed:
        changed = False
        best_order, best_t = None, route_time(order, d_sec, wh_idx)
        for i in range(n):
            for j in range(i + 1, n):
                cand = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                t = route_time(cand, d_sec, wh_idx)
                if t < best_t:
                    best_t, best_order = t, cand
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = list(order)
                node = cand.pop(i)
                cand.insert(j, node)
                t = route_time(cand, d_sec, wh_idx)
                if t < best_t:
                    best_t, best_order = t, cand
        if best_order is not None:
            order = best_order
            changed = True
    return order


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
    if USE_2OPT:
        order = improve_order(order, d_sec, wh_idx)
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


def _feasible_trip(nodes, pallets, d_sec, wh_idx):
    """True if `nodes` can be ordered into a trip within capacity and time."""
    if sum(pallets[n] for n in nodes) > TRUCK_CAPACITY:
        return False
    order = improve_order(nodes, d_sec, wh_idx)
    tmin = route_time(order, d_sec, wh_idx) / 60.0 + UNLOAD_MIN_PER_PALLET * sum(pallets[n] for n in nodes)
    return tmin <= MAX_TRIP_MIN


def generate_exhaustive(active, pallets, d_sec, wh_idx, name_of, pool, weekday):
    """Add EVERY feasible trip with 1-4 stores (checked on the best ordering
    found by 2-opt), making the pool complete up to 4 stops."""
    n = len(active)
    for i in active:
        add_trip(pool, [i], [i], pallets, d_sec, wh_idx, name_of, weekday)

    for x in range(n):
        a = active[x]
        for y in range(x + 1, n):
            b = active[y]
            if not _feasible_trip([a, b], pallets, d_sec, wh_idx):
                continue
            order = improve_order([a, b], d_sec, wh_idx)
            add_trip(pool, [a, b], order, pallets, d_sec, wh_idx, name_of, weekday)

    for x in range(n):
        a = active[x]
        for y in range(x + 1, n):
            b = active[y]
            for z in range(y + 1, n):
                c = active[z]
                if not _feasible_trip([a, b, c], pallets, d_sec, wh_idx):
                    continue
                order = improve_order([a, b, c], d_sec, wh_idx)
                add_trip(pool, [a, b, c], order, pallets, d_sec, wh_idx, name_of, weekday)

    for x in range(n):
        a = active[x]
        for y in range(x + 1, n):
            b = active[y]
            for z in range(y + 1, n):
                c = active[z]
                for w in range(z + 1, n):
                    e = active[w]
                    if not _feasible_trip([a, b, c, e], pallets, d_sec, wh_idx):
                        continue
                    order = improve_order([a, b, c, e], d_sec, wh_idx)
                    add_trip(pool, [a, b, c, e], order, pallets, d_sec, wh_idx, name_of, weekday)


def prune_dominated(pool):
    """Drop trips that cost more than serving their stores on separate trips.
    Such a trip can never appear in an optimal partition (the single-store
    trips are always available and cover the same stores more cheaply)."""
    single = {k[0]: v["TotalCost"] for k, v in pool.items() if len(k) == 1}
    return {k: v for k, v in pool.items()
            if v["TotalCost"] <= sum(single[s] for s in k) + 1e-6}


def generate_all(out_dir, use_2opt, rng, tag, exhaustive=False, prune=False):
    """Build the pool for every weekday into `out_dir`; returns {Weekday: DataFrame}."""
    global USE_2OPT
    USE_2OPT = use_2opt
    os.makedirs(out_dir, exist_ok=True)

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

    pools = {}

    for weekday in WEEKDAYS:
        pallets = est[weekday].to_numpy(dtype=int)
        active = [i for i, p in enumerate(pallets) if p > 0]
        pool = {}

        if exhaustive:
            generate_exhaustive(active, pallets, d_sec, wh_idx, name_of, pool, weekday)
        else:
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

        if prune:
            pool = prune_dominated(pool)

        rows = sorted(pool.values(), key=lambda r: r["TotalCost"])[:MAX_POOL_PER_DAY]

        covered = set()
        for r in rows:
            covered |= set(r["Stores"].split(","))
        uncovered = [store_names[i] for i in active if store_names[i] not in covered]

        df = pd.DataFrame(rows)
        df.insert(0, "RouteID", [f"{weekday[:3]}_{i:05d}" for i in range(1, len(df) + 1)])
        df.to_csv(os.path.join(out_dir, f"pool_{weekday}.csv"), index=False)

        hist = df["NStops"].value_counts().sort_index()
        ot_count = (df["OvertimeMin"] > 0).sum()
        print(f"[{tag}] {weekday}: {len(df)} unique trips | coverage "
              f"{100 * (len(active) - len(uncovered)) / len(active):.0f}% ({len(active)} stores)"
              + (f" | UNCOVERED: {uncovered}" if uncovered else ""))
        print(f"   stop histogram: " + ", ".join(f"{k}stop={int(v)}" for k, v in hist.items()))
        print(f"   overtime trips: {ot_count} | min trip cost ${df['TotalCost'].min():.2f}, "
              f"max ${df['TotalCost'].max():.2f}")

        pools[weekday] = df

    return pools


def main():
    v1 = generate_all("routes", False, np.random.default_rng(263), "v1")
    v2 = generate_all("routes_2opt", True, np.random.default_rng(263), "v2")
    v3 = generate_all("v3", True, np.random.default_rng(263), "v3", exhaustive=True, prune=True)

    print("\nPool comparison (drive min / pool cost $ / trips):")
    print(f"  {'weekday':10s} {'v1':>30s} {'v2':>30s} {'v3':>30s}")
    for weekday in WEEKDAYS:
        cells = []
        for v in (v1, v2, v3):
            drive = int(v[weekday]["DriveMin"].sum())
            cost = int(v[weekday]["TotalCost"].sum())
            cells.append(f"{drive}min / ${cost} / {len(v[weekday])}")
        print(f"  {weekday:10s} " + "  ".join(f"{c:>30s}" for c in cells))


if __name__ == "__main__":
    main()
