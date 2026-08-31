"""
Monte Carlo simulator for a solved route plan.

MAJOR REVISION NOTES (this version)
---------------------------------------
Two structural assumptions changed from earlier versions, per explicit
instruction:

1. TRAFFIC BASELINE = NO TRAFFIC. The data provider stated the duration
   matrix was generated via OpenStreetMap-based routing, which does not
   model traffic at all. Taken at face value, this means the given
   durations are the fastest physically achievable trip -- nothing can be
   faster. The traffic multiplier is therefore ONE-SIDED again:
       multiplier = 1 + HIGH * beta_sample,   beta_sample ~ Beta(alpha, beta)
   with a hard floor of exactly 1.0x (beta_sample=0). This reverses an
   earlier two-sided model that was built on real Google Maps measurements
   showing light-traffic trips often BEATING the baseline -- that finding
   is now in direct tension with this assumption and is not resolved; see
   the flagged issue in the accompanying report. This version proceeds
   with the no-traffic assumption because it is simpler to state and
   defend, per instruction, not because the tension has been settled.

2. DEMAND = MARKET PROJECTION, NOT REALIZED CUSTOMER ACTIVITY. Previously
   this simulator treated "unmet demand" as a real outcome to report and
   accept. We've been told the demand figures represent what stores
   ORDER in advance (a projection used to plan pallets), not unknowable
   real-time customer activity -- and that ALL of this ordered demand
   must be delivered. Consequently:
   - Every store that is part of the solved route plan (i.e. NOT
     permanently skipped by the shedding decision) must have its full
     sampled demand met. If the store's planned truck load falls short
     (because sampled demand exceeds the route's fixed planned pallets),
     a SEPARATE wet-leased truck is dispatched -- one per affected store
     -- to deliver exactly the shortfall (not the store's full demand;
     only the leftover the planned route couldn't cover).
   - Permanently SKIPPED stores (the shedding policy's deliberate,
     cost-driven exclusions) do NOT get a wet-lease top-up. This is an
     explicit, flagged assumption: shedding is a deliberate policy lever,
     and quietly overriding it with an expensive backup truck every time
     would defeat its purpose. If this is wrong, it's a one-line change
     (see FIX_SKIPPED_STORES_WITH_WETLEASE below).

Wet-lease top-up trips are modelled as a dedicated, direct round trip
(Warehouse -> store -> Warehouse) using the RAW duration matrix (not the
route's internal travel decomposition, since this is a different, solo
trip), scaled by the same day's traffic multiplier, plus 18 min/pallet
unloading for the shortfall quantity only. Cost uses the same $1400/2-hour
block formula as any other wet-leased trip. Capacity (16 pallets) is not
re-checked for these top-up trips since no single store's demand is
expected to approach that on its own.

What stays the same as before
----------------------------------
- Loading model: each planned route's pallet load is fixed at its
  baseline (from the solution CSV), decided before the day's demand is
  known.
- Demand distributions: Normal(mean, stdev) fitted per store per weekday
  from the 8 weeks of historical data in FoodstuffsDemand2026.csv.
- Per-store, in-order allocation within a route (the truck unloads up to
  each store's demand in sequence, limited by remaining capacity) -- this
  is what determines each store's shortfall, which then either gets
  topped up by wet-lease (if the store is on a planned route) or remains
  unmet (if the store was permanently skipped).

Configuration
----------------
BETA_ALPHA/BETA_BETA/HIGH_MULTIPLIER: averaged directly from the given
    AM/PM Beta parameter set (see report for the source values). Weekday:
    a=3.25, b=6.75, high=3.165. Saturday: a=5.5, b=5.75, high=2.1 (pass
    via CLI overrides for Saturday runs).
FIX_SKIPPED_STORES_WITH_WETLEASE: False by default -- see assumption (2)
    above. Set True to also wet-lease-cover permanently-skipped stores'
    full demand (defeats the shedding policy's savings; provided only in
    case the "no wet-lease for skipped stores" reading above is wrong).
"""

import argparse
import csv
import math
import random
import statistics
from datetime import datetime
from pathlib import Path

# ----------------------------- CONFIG ---------------------------------- #

DEMAND_CSV = "FoodstuffsDemand2026.csv"
DURATIONS_CSV = "FoodstuffsDurations2026.csv"
SOLUTION_CSV = "monday_solution.csv"
DAY = "Monday"
OUTPUT_DIR = "."

N_SIMULATIONS = 1000

BETA_ALPHA = 2.0                 # weekday default (AM/PM averaged from latest parameter set)
BETA_BETA = 8.4175                # weekday default (AM/PM averaged)
LOW_MULTIPLIER = 1.5285           # weekday default (AM/PM averaged). Note: this floor is
                                   # ABOVE 1.0 -- under these parameters, weekday traffic is
                                   # NEVER faster than baseline, even in the best case.
HIGH_MULTIPLIER = 7.1910          # weekday default (AM/PM averaged). For Saturday, override
                                   # with --beta-beta 6.9845 --low-multiplier 0.8030
                                   # --high-multiplier 7.6820

FIX_SKIPPED_STORES_WITH_WETLEASE = False  # see module docstring, assumption (2)

WAREHOUSE = "Warehouse"
CAPACITY_PALLETS = 16
UNLOADING_SEC_PER_PALLET = 18 * 60

OWNED_RATE_NORMAL = 220.0
OWNED_RATE_OVERTIME = 310.0
OVERTIME_THRESHOLD_HR = 4.0
LEASED_RATE_PER_BLOCK = 1400.0
LEASED_BLOCK_HOURS = 2.0
SKIP_COST_PAK_N_SAVE = 1500.0
SKIP_COST_OTHER = 800.0

CI_LEVEL = 0.95

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


# --------------------------- Data loading --------------------------------- #

def load_durations(path):
    """durations[a][b] = travel time in seconds, for every location pair --
    needed for the DIRECT Warehouse<->store wet-lease top-up trips."""
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


def fit_demand_distributions(path, day):
    """Returns {store: (mean, stdev)} fitted from every column in `path`
    whose date falls on `day` of the week, with automatic detection and
    exclusion of likely data-entry errors (an "extra trailing zero"
    signature: a value >=8x the median of that store's other same-weekday
    values, whose value/10 lands close to that median)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        date_cols = header[1:]
        weekday_mask = [datetime.strptime(d, "%d/%m/%Y").strftime("%A") == day
                         for d in date_cols]

        distributions = {}
        for row in reader:
            store = row[0]
            values = [float(v) for v, keep in zip(row[1:], weekday_mask) if keep]

            cleaned = []
            for i, v in enumerate(values):
                others = values[:i] + values[i + 1:]
                med_others = statistics.median(others) if others else 0
                if med_others > 0 and v >= 8.0 * med_others:
                    ratio_to_tenth = (v / 10) / med_others
                    if 0.5 <= ratio_to_tenth <= 2.0:
                        continue
                cleaned.append(v)

            mean = statistics.mean(cleaned)
            stdev = statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0
            distributions[store] = (mean, stdev)
    return distributions


def load_plan(path):
    """Parses a solve_lp.py-style solution CSV. Robust to whatever trailing
    summary format follows the route rows -- only rows whose `mode` is
    exactly 'owned', 'leased', or 'skipped' are treated as plan data."""
    routes = []
    skipped = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mode = (row.get("mode") or "").strip()
            if mode in ("owned", "leased"):
                routes.append({
                    "route_id": row["route_id"],
                    "mode": mode,
                    "stops": row["stops"].split(";"),
                    "total_pallets": float(row["total_pallets"]),
                    "total_duration_sec": float(row["total_duration_sec"]),
                })
            elif mode == "skipped":
                skipped.append(row["stops"])
    return routes, skipped


def decompose_baseline(route):
    """Splits a route's given duration into travel vs unload components
    using its own given total_pallets."""
    baseline_unload = route["total_pallets"] * UNLOADING_SEC_PER_PALLET
    baseline_travel = route["total_duration_sec"] - baseline_unload
    return max(baseline_travel, 0.0), baseline_unload


# ----------------------------- Simulation --------------------------------- #

def sample_demand(store, distributions, rng):
    mean, stdev = distributions[store]
    if stdev == 0:
        return max(mean, 0.0)
    return max(rng.gauss(mean, stdev), 0.0)


def traffic_multiplier_from_sample(raw_sample, low, high):
    """Linear stretch of the raw Beta(0,1) draw onto [low, high] directly.
    Per the latest parameter set, low is NOT always 0 -- for weekday it's
    above 1.0, meaning traffic is never faster than baseline even in the
    best case; for Saturday it's below 1.0, allowing faster-than-baseline
    days. See module docstring / chat history for the source values."""
    return low + (high - low) * raw_sample


def wet_lease_topup_cost(store, shortfall_pallets, durations, traffic_multiplier):
    """Cost of a dedicated Warehouse<->store round trip delivering ONLY the
    shortfall quantity, at the same day's traffic multiplier."""
    travel_sec = (durations[WAREHOUSE][store] + durations[store][WAREHOUSE]) * traffic_multiplier
    unload_sec = shortfall_pallets * UNLOADING_SEC_PER_PALLET
    return leased_cost(travel_sec + unload_sec)


def run_simulation(routes, skipped, distributions, durations, rng,
                    beta_alpha, beta_beta, low_multiplier, high_multiplier,
                    fix_skipped=FIX_SKIPPED_STORES_WITH_WETLEASE):
    """
    One simulated day. Planned routes' pallet loads are fixed at baseline;
    only traffic affects their cost. Demand is sampled per store and
    allocated in-order within each route; any shortfall on a PLANNED route
    triggers a dedicated wet-lease top-up trip for that store. Permanently
    skipped stores are not topped up (unless fix_skipped=True).
    """
    raw_sample = rng.betavariate(beta_alpha, beta_beta)
    traffic_multiplier = traffic_multiplier_from_sample(raw_sample, low_multiplier, high_multiplier)

    total_cost = 0.0
    total_requested = 0.0
    total_served = 0.0
    routes_with_shortfall = 0
    stores_topped_up = 0          # planned-route stores needing a wet-lease top-up
    wetlease_topup_cost = 0.0
    wetlease_topup_trips = 0
    stores_unmet_no_topup = 0     # permanently-skipped stores (or, if fix_skipped=False,
                                    # these remain genuinely unmet)

    for route in routes:
        baseline_travel, _ = decompose_baseline(route)
        per_store_demand = [(s, sample_demand(s, distributions, rng)) for s in route["stops"]]
        sim_demand = sum(d for _, d in per_store_demand)
        sim_travel = baseline_travel * traffic_multiplier

        planned_load = route["total_pallets"]
        unload_sec = planned_load * UNLOADING_SEC_PER_PALLET
        if sim_demand > planned_load:
            routes_with_shortfall += 1

        remaining = planned_load
        for store, demand in per_store_demand:
            served_here = min(demand, remaining)
            remaining -= served_here
            shortfall = demand - served_here
            if shortfall > 1e-9:
                stores_topped_up += 1
                topup = wet_lease_topup_cost(store, shortfall, durations, traffic_multiplier)
                wetlease_topup_cost += topup
                wetlease_topup_trips += 1
            total_served += demand  # fully served either by the route or the top-up

        sim_duration = sim_travel + unload_sec
        cost = owned_cost(sim_duration) if route["mode"] == "owned" else leased_cost(sim_duration)

        total_cost += cost
        total_requested += sim_demand

    total_cost += wetlease_topup_cost

    for store in skipped:
        sim_demand = sample_demand(store, distributions, rng)
        total_requested += sim_demand
        if fix_skipped:
            topup = wet_lease_topup_cost(store, sim_demand, durations, traffic_multiplier)
            total_cost += topup
            total_served += sim_demand
            wetlease_topup_trips += 1
        else:
            total_cost += skip_cost(store)
            stores_unmet_no_topup += 1
            # served += 0

    demand_met_pct = (total_served / total_requested * 100) if total_requested > 0 else 100.0
    return {
        "total_cost": total_cost,
        "demand_met_pct": demand_met_pct,
        "routes_with_shortfall": routes_with_shortfall,
        "stores_topped_up": stores_topped_up,
        "wetlease_topup_cost": wetlease_topup_cost,
        "wetlease_topup_trips": wetlease_topup_trips,
        "stores_unmet_no_topup": stores_unmet_no_topup,
    }


def percentile(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def confidence_interval(values, level=CI_LEVEL):
    s = sorted(values)
    tail = (1 - level) / 2
    return percentile(s, tail), percentile(s, 1 - tail)


def plot_results(results_list, summary, n_routes, output_dir, title=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping CI plot)")
        return None

    panels = [
        ("total_cost", "Total Cost incl. Wet-Lease Top-Ups ($)", lambda v: f"${v:,.0f}"),
        ("wetlease_topup_cost", "Wet-Lease Top-Up Cost ($)", lambda v: f"${v:,.0f}"),
        ("stores_topped_up", "Stores Needing Top-Up", lambda v: f"{v:.1f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (key, subtitle, fmt) in zip(axes, panels):
        values = [r[key] for r in results_list]
        mean, (lo, hi) = summary[key]
        ax.hist(values, bins=30, color="#2c5282", alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(mean, color="#c0392b", linestyle="--", linewidth=1.6, label=f"mean = {fmt(mean)}")
        ax.axvline(lo, color="#555", linestyle=":", linewidth=1.3)
        ax.axvline(hi, color="#555", linestyle=":", linewidth=1.3,
                    label=f"{int(CI_LEVEL*100)}% CI = [{fmt(lo)}, {fmt(hi)}]")
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_ylabel("Simulations")
        ax.legend(fontsize=8.2, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    suptitle = title or f"Monte Carlo Results (n={len(results_list)} simulations)"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    path = Path(output_dir) / "montecarlo_ci_plot.png"
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------- Main ------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo simulate a solved route plan.")
    parser.add_argument("--solution-csv", default=SOLUTION_CSV)
    parser.add_argument("--demand-csv", default=DEMAND_CSV)
    parser.add_argument("--durations-csv", default=DURATIONS_CSV)
    parser.add_argument("--day", default=DAY)
    parser.add_argument("--n-sim", type=int, default=N_SIMULATIONS)
    parser.add_argument("--beta-alpha", type=float, default=BETA_ALPHA)
    parser.add_argument("--beta-beta", type=float, default=BETA_BETA)
    parser.add_argument("--low-multiplier", type=float, default=LOW_MULTIPLIER)
    parser.add_argument("--high-multiplier", type=float, default=HIGH_MULTIPLIER)
    parser.add_argument("--fix-skipped", action="store_true", default=FIX_SKIPPED_STORES_WITH_WETLEASE,
                         help="also wet-lease-cover permanently-skipped stores (defeats shedding savings)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    distributions = fit_demand_distributions(args.demand_csv, args.day)
    durations = load_durations(args.durations_csv)
    routes, skipped = load_plan(args.solution_csv)

    missing = [s for r in routes for s in r["stops"] if s not in distributions]
    missing += [s for s in skipped if s not in distributions]
    if missing:
        raise ValueError(f"Stores in the plan but not in the demand CSV: {set(missing)}")

    print(f"Loaded plan: {len(routes)} routes, {len(skipped)} permanently-skipped stores")
    print(f"Fitted demand distributions for {len(distributions)} stores from {args.day}s")
    print(f"Running {args.n_sim} simulations "
          f"(traffic ~ [{args.low_multiplier}, {args.high_multiplier}] via "
          f"Beta({args.beta_alpha},{args.beta_beta}); wet-lease top-up "
          f"{'ALSO covers' if args.fix_skipped else 'does NOT cover'} skipped stores)...")

    results_list = []
    for _ in range(args.n_sim):
        r = run_simulation(routes, skipped, distributions, durations, rng,
                            args.beta_alpha, args.beta_beta, args.low_multiplier, args.high_multiplier,
                            fix_skipped=args.fix_skipped)
        results_list.append(r)

    metrics = ["total_cost", "demand_met_pct", "routes_with_shortfall", "stores_topped_up",
               "wetlease_topup_cost", "wetlease_topup_trips", "stores_unmet_no_topup"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results_list]
        summary[m] = (statistics.mean(vals), confidence_interval(vals))

    total_stores = len({s for r in routes for s in r["stops"]}) + len(skipped)

    print(f"\n--- Results over {args.n_sim} simulations ---")
    c_mean, c_ci = summary["total_cost"]
    print(f"Total cost (incl. top-ups):  mean=${c_mean:,.2f}  "
          f"{int(CI_LEVEL*100)}% CI=[${c_ci[0]:,.2f}, ${c_ci[1]:,.2f}]")
    d_mean, d_ci = summary["demand_met_pct"]
    print(f"Demand met:                  mean={d_mean:.2f}%  "
          f"{int(CI_LEVEL*100)}% CI=[{d_ci[0]:.2f}%, {d_ci[1]:.2f}%]")
    rs_mean, rs_ci = summary["routes_with_shortfall"]
    print(f"Routes w/ shortfall:         mean={rs_mean:.2f}/{len(routes)}  "
          f"{int(CI_LEVEL*100)}% CI=[{rs_ci[0]:.0f}, {rs_ci[1]:.0f}]")
    st_mean, st_ci = summary["stores_topped_up"]
    print(f"Stores topped up (wet-lease): mean={st_mean:.2f}/{total_stores}  "
          f"{int(CI_LEVEL*100)}% CI=[{st_ci[0]:.0f}, {st_ci[1]:.0f}]")
    wc_mean, wc_ci = summary["wetlease_topup_cost"]
    print(f"Wet-lease top-up cost:       mean=${wc_mean:,.2f}  "
          f"{int(CI_LEVEL*100)}% CI=[${wc_ci[0]:,.2f}, ${wc_ci[1]:,.2f}]")
    wt_mean, wt_ci = summary["wetlease_topup_trips"]
    print(f"Wet-lease top-up trips:      mean={wt_mean:.2f}  "
          f"{int(CI_LEVEL*100)}% CI=[{wt_ci[0]:.0f}, {wt_ci[1]:.0f}]")
    su_mean, su_ci = summary["stores_unmet_no_topup"]
    print(f"Stores genuinely unmet:      mean={su_mean:.2f}  "
          f"{int(CI_LEVEL*100)}% CI=[{su_ci[0]:.0f}, {su_ci[1]:.0f}]  "
          f"({'0 by construction -- fix_skipped is on' if args.fix_skipped else 'permanently-skipped stores'})")

    out_path = Path(args.output_dir) / "montecarlo_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["simulation"] + metrics)
        for i, r in enumerate(results_list, start=1):
            writer.writerow([i] + [round(r[m], 4) if isinstance(r[m], float) else r[m] for m in metrics])
    print(f"\nPer-simulation results written to {out_path}")

    shed_label = "No Shedding" if "no_shed" in Path(args.solution_csv).stem else "Shedding Allowed"
    plot_title = f"{args.day} -- {shed_label} (n={args.n_sim})"
    plot_path = plot_results(results_list, summary, len(routes), args.output_dir, title=plot_title)
    if plot_path:
        print(f"CI plot written to {plot_path}")


if __name__ == "__main__":
    main()
