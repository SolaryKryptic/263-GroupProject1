"""
Set-covering MIP for the Foodstuffs trucking problem -- picks the lowest-
cost combination of candidate routes (from the route generator's output)
to serve a single day's demand.

Decision variables (all binary), per candidate route r and store s:
    x_r  = 1 if route r runs on an OWNED Foodstuffs truck
    v_r  = 1 if route r runs on a WET-LEASED (Linfox) truck
    z_s  = 1 if store s's delivery is SKIPPED that day

Objective:  minimize  sum(owned_cost(r)*x_r) + sum(leased_cost(r)*v_r)
                       + sum(skip_cost(s)*z_s)

Constraints:
    Coverage (exact, no splitting a store's demand across two trucks):
        sum_{r covers s} (x_r + v_r) + z_s == 1          for every store s
    One mode per route:
        x_r + v_r <= 1                                    for every route r
    Fleet capacity (20 trucks x 2 shifts/day):
        sum(x_r) <= 40
    Demand-shedding cap (fuel-reduction proposal, part 3.1):
        sum(z_s) <= 0.20 * num_stores      -- only active when --allow-skip
        sum(z_s) == 0                      -- baseline model, --no-skip

Cost functions:
    owned_cost(r)  = $220/hr for the first 4 hours of route duration,
                     $310/hr for any time beyond 4 hours (piecewise, but
                     since route duration is a fixed precomputed number for
                     every candidate route -- not a decision variable --
                     this needs no special piecewise-MIP machinery; it's
                     just evaluated in Python and handed to the solver as
                     an ordinary constant coefficient).
    leased_cost(r) = $1400 per 2-hour block (any part-block rounds up).
    skip_cost(s)   = $1500 for a Pak 'n Save, $800 for any other chain.

Store types (Pak 'n Save vs other) are read directly off the store names
in the routes CSV's `stops` column -- no separate demand file is needed,
since the route generator has already baked pallet totals into each
candidate route.

Usage
-----
    python3 solve_lp.py Monday                       # solve here directly
    python3 solve_lp.py Monday --export-lp            # write monday_model.lp instead
                                                        # (import into Excel via OpenSolver's
                                                        # "Import LP file" feature and solve there)
    python3 solve_lp.py Saturday --routes-csv routes_Saturday.csv
    python3 solve_lp.py Monday --no-skip               # baseline model: no fuel-reduction
                                                        # shedding allowed (comparison run
                                                        # for part 3.1)
    python3 solve_lp.py --all                          # solve every routes_<Day>.csv found
                                                        # in OUTPUT_DIR and print a summary

Requires the routes_<Day>.csv files (route_id, stops, num_stops,
total_pallets, total_duration_sec, ...) to already exist in OUTPUT_DIR.
"""

import argparse
import csv
import math
from pathlib import Path

import pulp

# ----------------------------- CONFIG ---------------------------------- #

OUTPUT_DIR = "."
DEMAND_CSV = "ayush_estimates.csv"  # optional -- Supermarket,Mon..Sun,Type,Total

OWNED_RATE_NORMAL = 220.0      # $/hr, up to the overtime threshold
OWNED_RATE_OVERTIME = 310.0    # $/hr, beyond the overtime threshold
OVERTIME_THRESHOLD_HR = 4.0

LEASED_RATE_PER_BLOCK = 1400.0
LEASED_BLOCK_HOURS = 2.0

SKIP_COST_PAK_N_SAVE = 1500.0
SKIP_COST_OTHER = 800.0

FLEET_SIZE = 20
SHIFTS_PER_TRUCK_PER_DAY = 2
MAX_OWNED_TRIPS = FLEET_SIZE * SHIFTS_PER_TRUCK_PER_DAY   # 40

MAX_SKIP_FRACTION = 0.20

# Solver-tractability guard: if the route pool is huge, keep only the N
# most pallet-efficient (by TRUE dollar cost, not raw seconds) routes per
# store before handing the problem to CBC. Set to None to disable.
MAX_ROUTES_PER_STORE = 500
TRIM_THRESHOLD = 15000  # only trim if the pool is bigger than this

ALL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# ------------------------------------------------------------------------ #


def owned_cost(duration_sec):
    hours = duration_sec / 3600.0
    normal = min(hours, OVERTIME_THRESHOLD_HR)
    overtime = max(hours - OVERTIME_THRESHOLD_HR, 0.0)
    return OWNED_RATE_NORMAL * normal + OWNED_RATE_OVERTIME * overtime


def leased_cost(duration_sec):
    hours = duration_sec / 3600.0
    return LEASED_RATE_PER_BLOCK * math.ceil(hours / LEASED_BLOCK_HOURS)


def skip_cost(store_name):
    return SKIP_COST_PAK_N_SAVE if "Pak 'n Save" in store_name else SKIP_COST_OTHER


def load_routes(path):
    routes = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            routes.append({
                "route_id": row["route_id"],
                "stops": row["stops"].split(";"),
                "num_stops": int(row["num_stops"]),
                "total_pallets": int(row["total_pallets"]),
                "total_duration_sec": float(row["total_duration_sec"]),
            })
    return routes


def store_types_from_routes(active_stores):
    """Classify every store touched by the day's candidate routes as
    Pak 'n Save or Other, straight off the store name -- fallback used
    when no demand CSV is available."""
    return {s: ("Pak" if "Pak 'n Save" in s else "Other") for s in active_stores}


def load_demand(path):
    """Loads the Part-1 pallet demand estimates
    (Supermarket,Monday,...,Saturday,Sunday,Type,Total). Returns
    {store: {"Type": str, day: pallets, ...}}, or None if the file
    isn't present -- callers fall back to name-based type inference."""
    p = Path(path)
    if not p.exists():
        return None
    demand = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            store = row["Supermarket"]
            demand[store] = row
    return demand


def store_types_from_demand(demand, active_stores):
    """Authoritative store typing straight from the demand file's Type
    column (more robust than string-matching, e.g. keeps 'New World
    Metro' distinct from plain 'New World' if that ever matters)."""
    types = {}
    for s in active_stores:
        row = demand.get(s)
        if row is None:
            types[s] = "Pak" if "Pak 'n Save" in s else "Other"
        else:
            types[s] = "Pak" if row["Type"] == "Pak 'n Save" else "Other"
    return types


def demand_pallets_for_day(demand, store, day):
    """Looks up the Part-1 estimated pallet count for one store on one
    weekday. Returns None if unavailable."""
    if demand is None or store not in demand:
        return None
    val = demand[store].get(day)
    return int(val) if val not in (None, "") else None


def trim_by_dollar_cost(routes, max_per_store):
    """Keep only the max_per_store cheapest (by true $/pallet) routes
    touching each store -- caps problem size without distorting quality,
    since ranking is by actual dollar cost, not a raw-time proxy that
    would unfairly penalize longer, well-packed routes."""
    from collections import defaultdict
    by_store = defaultdict(list)
    for r in routes:
        for s in r["stops"]:
            by_store[s].append(r)
    keep_ids = set()
    for store, rs in by_store.items():
        rs.sort(key=lambda r: owned_cost(r["total_duration_sec"]) / r["total_pallets"])
        for r in rs[:max_per_store]:
            keep_ids.add(tuple(r["stops"]))
    return [r for r in routes if tuple(r["stops"]) in keep_ids]


def build_model(routes, store_types, allow_skip=True):
    """Builds the PuLP model (variables, objective, constraints) without
    solving it -- shared by both solve() and export_lp().

    allow_skip=True  -> fuel-reduction proposal: up to 20% of stores may
                         be skipped (skip_cap constraint, part 3.1).
    allow_skip=False -> baseline model: every store must be served, so
                         this is the "no fuel reduction" comparison case.
    """
    all_stores = sorted(store_types.keys())

    prob = pulp.LpProblem("day_routing", pulp.LpMinimize)
    x = {r["route_id"]: pulp.LpVariable(f"own_{r['route_id']}", cat="Binary") for r in routes}
    v = {r["route_id"]: pulp.LpVariable(f"lease_{r['route_id']}", cat="Binary") for r in routes}
    z = {s: pulp.LpVariable(f"skip_{i}", cat="Binary") for i, s in enumerate(all_stores)}

    oc = {r["route_id"]: owned_cost(r["total_duration_sec"]) for r in routes}
    lc = {r["route_id"]: leased_cost(r["total_duration_sec"]) for r in routes}

    prob += (pulp.lpSum(oc[rid] * x[rid] for rid in x)
             + pulp.lpSum(lc[rid] * v[rid] for rid in v)
             + pulp.lpSum(skip_cost(s) * z[s] for s in all_stores))

    routes_by_store = {s: [] for s in all_stores}
    for r in routes:
        for s in r["stops"]:
            routes_by_store[s].append(r["route_id"])

    for s in all_stores:
        prob += (pulp.lpSum(x[rid] + v[rid] for rid in routes_by_store[s]) + z[s] == 1,
                  f"cover_{s}")

    for r in routes:
        prob += x[r["route_id"]] + v[r["route_id"]] <= 1, f"one_mode_{r['route_id']}"

    prob += pulp.lpSum(x.values()) <= MAX_OWNED_TRIPS, "fleet_capacity"

    if allow_skip:
        prob += pulp.lpSum(z.values()) <= MAX_SKIP_FRACTION * len(all_stores), "skip_cap"
    else:
        prob += pulp.lpSum(z.values()) == 0, "no_skip_allowed"

    return prob, x, v, z, oc, lc


def export_lp(day, routes, store_types, allow_skip=True, output_dir=OUTPUT_DIR):
    """
    Writes the model to a standard CPLEX-LP-format .lp file, WITHOUT
    solving it. Open this in Excel via OpenSolver's "Import LP file"
    feature (Data tab -> OpenSolver -> Import Model -> LP file) to build
    and solve the model natively in Excel using OpenSolver's CBC backend.
    """
    prob, x, v, z, oc, lc = build_model(routes, store_types, allow_skip=allow_skip)
    suffix = "" if allow_skip else "_noskip"
    path = Path(output_dir) / f"{day.lower()}_model{suffix}.lp"
    prob.writeLP(str(path))
    print(f"LP model written to {path}")
    print(f"  Variables: {len(x) + len(v) + len(z)}  "
          f"({len(x)} owned-route + {len(v)} leased-route + {len(z)} skip)")
    print(f"  Constraints: {len(store_types) + len(routes) + 2}  "
          f"({len(store_types)} coverage + {len(routes)} one-mode-per-route + fleet-cap + skip-cap)")
    print("In Excel: Data tab -> OpenSolver -> Model (or 'Import Model') -> "
          "choose 'LP file' -> select this file.")
    return path


def solve(routes, store_types, allow_skip=True):
    prob, x, v, z, oc, lc = build_model(routes, store_types, allow_skip=allow_skip)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    owned_routes = [r for r in routes if x[r["route_id"]].value() > 0.5]
    leased_routes = [r for r in routes if v[r["route_id"]].value() > 0.5]
    skipped = [s for s in store_types if z[s].value() > 0.5]

    return status, owned_routes, leased_routes, skipped, oc, lc


def report_and_save(day, status, owned_routes, leased_routes, skipped, oc, lc,
                     allow_skip=True, output_dir=OUTPUT_DIR, demand=None):
    total_owned = sum(oc[r["route_id"]] for r in owned_routes)
    total_leased = sum(lc[r["route_id"]] for r in leased_routes)
    total_skip = sum(skip_cost(s) for s in skipped)
    grand_total = total_owned + total_leased + total_skip

    label = f"{day} ({'fuel-reduction allowed' if allow_skip else 'baseline, no skipping'})"
    print(f"\n{label} -- Solver status: {status}")
    print(f"Owned-fleet trips: {len(owned_routes)}/{MAX_OWNED_TRIPS}   "
          f"Wet-leased trips: {len(leased_routes)}   Skipped stores: {len(skipped)}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    print(f"Owned cost:  ${total_owned:,.2f}")
    print(f"Leased cost: ${total_leased:,.2f}")
    print(f"Skip cost:   ${total_skip:,.2f}")
    print(f"TOTAL COST:  ${grand_total:,.2f}")

    # Pallet reconciliation against the Part-1 demand estimate, if available.
    if demand is not None:
        served_pallets = sum(r["total_pallets"] for r in owned_routes + leased_routes)
        skipped_pallets = sum(demand_pallets_for_day(demand, s, day) or 0 for s in skipped)
        total_demand = served_pallets + skipped_pallets
        if total_demand > 0:
            print(f"Pallets served: {served_pallets}   Pallets skipped: {skipped_pallets}   "
                  f"({skipped_pallets / total_demand:.1%} of demand shed)")

    suffix = "" if allow_skip else "_noskip"
    path = Path(output_dir) / f"{day.lower()}_solution{suffix}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["route_id", "mode", "stops", "num_stops", "total_pallets",
                          "total_duration_sec", "cost"])
        for r in owned_routes:
            writer.writerow([r["route_id"], "owned", ";".join(r["stops"]), r["num_stops"],
                              r["total_pallets"], r["total_duration_sec"], oc[r["route_id"]]])
        for r in leased_routes:
            writer.writerow([r["route_id"], "leased", ";".join(r["stops"]), r["num_stops"],
                              r["total_pallets"], r["total_duration_sec"], lc[r["route_id"]]])
        for s in skipped:
            writer.writerow([f"SKIP_{s}", "skipped", s, 0, 0, 0, skip_cost(s)])
    print(f"Solution written to {path}")
    return grand_total


def run_one_day(day, routes_csv=None, no_trim=False, allow_skip=True,
                 export_lp_only=False, output_dir=OUTPUT_DIR):
    routes_path = routes_csv or f"{output_dir}/routes_{day}.csv"
    routes = load_routes(routes_path)
    active_stores = {s for r in routes for s in r["stops"]}

    demand = load_demand(f"{output_dir}/{DEMAND_CSV}")
    if demand is not None:
        store_types = store_types_from_demand(demand, active_stores)
    else:
        store_types = store_types_from_routes(active_stores)

    if not no_trim and MAX_ROUTES_PER_STORE and len(routes) > TRIM_THRESHOLD:
        routes = trim_by_dollar_cost(routes, MAX_ROUTES_PER_STORE)
        print(f"Pool trimmed to {len(routes)} routes for solver tractability.")

    print(f"Loaded {len(routes)} candidate routes covering {len(active_stores)} stores.")

    if export_lp_only:
        export_lp(day, routes, store_types, allow_skip=allow_skip, output_dir=output_dir)
        return None

    status, owned_routes, leased_routes, skipped, oc, lc = solve(
        routes, store_types, allow_skip=allow_skip)
    return report_and_save(day, status, owned_routes, leased_routes, skipped, oc, lc,
                            allow_skip=allow_skip, output_dir=output_dir, demand=demand)


def main():
    parser = argparse.ArgumentParser(description="Solve the routing MIP for a given day.")
    parser.add_argument("day", nargs="?", help="e.g. Monday (omit if using --all)")
    parser.add_argument("--routes-csv", default=None,
                         help="path to the routes CSV (default: OUTPUT_DIR/routes_<Day>.csv)")
    parser.add_argument("--no-trim", action="store_true",
                         help="disable the solver-tractability trim, even for huge pools")
    parser.add_argument("--export-lp", action="store_true",
                         help="write the model to a .lp file instead of solving it here "
                              "(for OpenSolver's Import LP file feature in Excel)")
    parser.add_argument("--no-skip", action="store_true",
                         help="baseline model: disallow demand-shedding entirely (for "
                              "comparison against the fuel-reduction proposal)")
    parser.add_argument("--all", action="store_true",
                         help="solve every day in %s and print a weekly summary" % ALL_DAYS)
    args = parser.parse_args()

    allow_skip = not args.no_skip

    if args.all:
        if args.export_lp:
            parser.error("--all and --export-lp cannot be combined")
        totals = {}
        for day in ALL_DAYS:
            path = Path(OUTPUT_DIR) / f"routes_{day}.csv"
            if not path.exists():
                print(f"Skipping {day}: {path} not found.")
                continue
            totals[day] = run_one_day(day, no_trim=args.no_trim, allow_skip=allow_skip)
        print("\n=== Weekly summary ===")
        grand_total = 0.0
        for day, total in totals.items():
            print(f"  {day:10s} ${total:,.2f}")
            grand_total += total
        print(f"  {'TOTAL':10s} ${grand_total:,.2f}")
        return

    if not args.day:
        parser.error("day is required unless --all is given")

    run_one_day(args.day, routes_csv=args.routes_csv, no_trim=args.no_trim,
                allow_skip=allow_skip, export_lp_only=args.export_lp)


if __name__ == "__main__":
    main()
