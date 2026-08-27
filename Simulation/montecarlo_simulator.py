"""
Monte Carlo simulator for a solved route plan.

Answers two questions about an already-solved plan (from solve_lp.py's
output CSV): given that both traffic conditions and store demand are
uncertain in reality, how confident can we actually be in (a) the plan's
total cost, and (b) how much of real demand it ends up serving?

Loading model: pallet quantities are locked in at the plan's baseline
before the actual day's demand is known, and never adjusted -- there is
no live/updating demand-sensing system, so the truck cannot load "what's
needed today" because that isn't known at load-time. The only information
available when loading happens is the plan's baseline figure. Demand only
enters afterward, as a hindsight comparison of whether that fixed load
covered what was actually needed:
    planned_load = route's baseline total_pallets  (always <=16, since
                    the route was built respecting truck capacity)
    delivered    = min(planned_load, sim_demand)
    unload_sec   = planned_load * UNLOADING_SEC_PER_PALLET  (FIXED -- the
                    truck always carries and unloads this much, regardless
                    of what the day's demand turns out to be)
    sim_travel   = baseline_travel_sec * traffic_multiplier
Only TRAFFIC affects cost -- unload time never varies, since the load
itself never varies. Demand only affects the SEPARATE demand-met metric.

A "shortfall" (in the routes_with_shortfall count) means the day's actual
demand exceeded the PLANNED LOAD -- NOT a truck capacity breach (the plan
always respects the 16-pallet cap by construction). It means the fixed
plan simply didn't allocate that route enough of the truck's available
room for that particular day's demand.

Two sources of randomness per simulation:
    1. TRAFFIC -- one Beta-distributed multiplier per simulation, applied
       to every route's TRAVEL time that day (traffic is treated as a
       network-wide condition, not independent per route). A raw Beta
       draw (in [0,1]) is mapped LINEARLY onto [LOW_MULTIPLIER,
       HIGH_MULTIPLIER]:
           multiplier = LOW_MULTIPLIER + (HIGH_MULTIPLIER - LOW_MULTIPLIER) * beta_sample
       Deliberately simple -- an earlier version tried to force the Beta
       shape's peak onto exactly 1.0x via a piecewise rescaling, but that
       centering was an assumption imposed on the data, not something the
       Beta shape itself justified. Here the peak lands wherever the raw
       mode naturally falls within [LOW_MULTIPLIER, HIGH_MULTIPLIER] --
       nothing forces it to any particular value.
    2. DEMAND -- each store's pallet demand that day is drawn from a
       Normal distribution fitted directly from the 8 weeks of actual
       daily data in FoodstuffsDemand2026.csv (mean and stdev computed
       per store, per weekday -- fully data-driven, no manual tuning).

Known limitation worth flagging: trucks run two shifts/day (8am and 2pm),
and empirically these have DIFFERENT typical traffic (a real Google Maps
check found 2pm running noticeably lighter than the 8am-inclusive blended
weekday estimate this model currently uses). The solution CSVs don't
currently record which shift a given route runs in, so this model applies
one blended traffic distribution across all routes on a day rather than
distinguishing AM-dispatched from PM-dispatched routes. Revisit if
per-route shift assignment becomes available.

What stays FIXED across every simulation: which routes exist, which
stores they visit, each route's mode (owned / wet-leased), and the skip
decisions. This simulator asks "how did our ALREADY-CHOSEN plan perform
under uncertainty", not "would a different plan have been better".

How duration is recomputed per simulation
--------------------------------------------
Each route's ORIGINAL total_duration_sec (from the solution CSV) is split
into a travel component and an unload component using that route's own
GIVEN total_pallets (self-consistent -- doesn't require the demand CSV's
values to exactly match whatever was originally used to build the route):
    baseline_unload_sec = total_pallets * UNLOADING_SEC_PER_PALLET
    baseline_travel_sec = total_duration_sec - baseline_unload_sec

Configuration
----------------
BETA_ALPHA / BETA_BETA: shape of the traffic distribution.
LOW_MULTIPLIER: empirically grounded via 7 real Google Maps checks (2pm,
    light traffic) against this dataset's own durations for the same
    pairs -- mean ratio 0.869, median 0.866, range 0.72-1.09.
HIGH_MULTIPLIER: less rigorously grounded than LOW_MULTIPLIER -- currently
    set from the originally-given Beta parameter set's "high" value.
    Tune freely; the resulting peak/mean shift with it (see chat history
    for a table of HIGH -> peak/mean under these Beta shapes).
N_SIMULATIONS: number of Monte Carlo draws. Set to 1000 for now.

BETA_ALPHA / BETA_BETA calibration note: fixed at a=1.5 (right-skewed shape
-- see chat history for why volume-fitted, near-symmetric shapes were
rejected: nonlinear volume-to-delay transforms amplify skewness relative
to a raw traffic-volume distribution's own shape). b is then SOLVED, not
guessed, so that the resulting multiplier distribution's MEDIAN lands
exactly on 1.0x -- i.e. a genuine 50/50 split between faster- and slower-
than-baseline days. This was chosen as a defensible "neutral" assumption
given no strong evidence either way -- though note a real caveat: if the
given baseline durations represent a MEAN (not median) of historical
travel times, right-skewed real-world traffic would imply P(faster) should
be ABOVE 50%, not exactly at it, since the mean of a right-skewed
distribution sits above its median. We don't have enough information to
resolve this precisely, so 50/50 is a stated, flagged assumption, not a
proven value.

Demand distribution parameters (mean, stdev per store) are NOT manual
inputs -- they're fitted directly from FoodstuffsDemand2026.csv each run.
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
SOLUTION_CSV = "monday_solution.csv"
DAY = "Monday"                  # which weekday's columns to fit demand from
OUTPUT_DIR = "."

N_SIMULATIONS = 1000            # <-- adjust freely

# --- Traffic model (Beta distribution) -- PLACEHOLDERS, tune later -----
BETA_ALPHA = 1.5                # weekday default. a=1.5 gives a right-skewed shape (mode
                                 # pulled toward the good-traffic side); Saturday uses the
                                 # same alpha via --beta-alpha 1.5 (no change needed, it's
                                 # already the default) but a DIFFERENT beta -- see below.
BETA_BETA = 8.874                # weekday default. Solved (see calibration note below) so
                                 # the resulting multiplier's MEDIAN lands exactly on 1.0x --
                                 # i.e. a genuine 50/50 split between faster- and slower-
                                 # than-baseline days. For SATURDAY runs, override with
                                 # --beta-beta 4.671 (same 50/50-median logic, solved
                                 # separately since Saturday's [LOW,HIGH] range differs).
LOW_MULTIPLIER = 0.7            # NOTE: the empirically-grounded value from 7 real Google
                                 # Maps checks (2pm, light traffic) was actually 0.85 (mean
                                 # ratio 0.869, median 0.866, range 0.72-1.09 vs this
                                 # dataset's own durations for the same pairs). Lowered to
                                 # 0.7 deliberately for proportional symmetry with
                                 # HIGH_MULTIPLIER, which is a much larger, ungrounded swing
                                 # above 1.0 -- this is a stylistic/consistency choice, not
                                 # something the evidence itself supports. Revert to 0.85 to
                                 # go with the strongest available grounding.
HIGH_MULTIPLIER = 3.165         # <-- tune freely. Currently the originally-given Beta
                                 # parameter set's "high" value; not independently
                                 # validated the way the empirical LOW_MULTIPLIER estimate is.

# --- Fixed problem constants (must match the MILP that produced the plan) --
CAPACITY_PALLETS = 16
UNLOADING_SEC_PER_PALLET = 18 * 60

OWNED_RATE_NORMAL = 220.0
OWNED_RATE_OVERTIME = 310.0
OVERTIME_THRESHOLD_HR = 4.0
LEASED_RATE_PER_BLOCK = 1400.0
LEASED_BLOCK_HOURS = 2.0
SKIP_COST_PAK_N_SAVE = 1500.0
SKIP_COST_OTHER = 800.0

CI_LEVEL = 0.95                 # confidence interval width (percentile method)

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


# --------------------------- Demand fitting ------------------------------ #

def fit_demand_distributions(path, day, outlier_ratio=8.0):
    """
    Returns {store: (mean, stdev)} fitted from every column in `path`
    whose date falls on `day` of the week. stdev uses the sample formula
    (ddof=1, i.e. divides by n-1) since these 8 values are a sample of
    possible daily outcomes, not the full population.

    Outlier handling: with only ~8 historical points per store, a single
    corrupted value (e.g. a data-entry error with an extra trailing zero)
    can badly distort both the mean and stdev. Before fitting, any value
    that is both (a) at least `outlier_ratio`x the median of that store's
    OTHER same-weekday values, AND (b) whose value/10 lands close to that
    median (0.5x-2x) -- the specific signature of an accidental extra
    zero -- is excluded from the fit. Verified against the full dataset:
    this flags exactly 2 points (both on Wednesday, both matching the
    extra-zero pattern precisely) and nothing else.
    """
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

            # outlier detection -- see docstring
            cleaned = []
            for i, v in enumerate(values):
                others = values[:i] + values[i + 1:]
                med_others = statistics.median(others) if others else 0
                if med_others > 0 and v >= outlier_ratio * med_others:
                    ratio_to_tenth = (v / 10) / med_others
                    if 0.5 <= ratio_to_tenth <= 2.0:
                        continue  # exclude: looks like an extra trailing zero
                cleaned.append(v)

            mean = statistics.mean(cleaned)
            stdev = statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0
            distributions[store] = (mean, stdev)
    return distributions


# --------------------------- Plan parsing --------------------------------- #

def load_plan(path):
    """
    Parses a solve_lp.py-style solution CSV. Robust to whatever trailing
    summary format follows the route rows (a single TOTAL row, a full
    SUMMARY block, or nothing) -- only rows whose `mode` is exactly
    'owned', 'leased', or 'skipped' are treated as plan data; anything
    else (blank lines, TOTAL rows, summary key/value rows) is ignored.
    """
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
                skipped.append(row["stops"])  # single store name for skip rows
    return routes, skipped


def decompose_baseline(route):
    """Splits a route's given duration into travel vs unload components
    using its OWN given total_pallets (self-consistent, no dependency on
    matching an external demand source's exact values)."""
    baseline_unload = route["total_pallets"] * UNLOADING_SEC_PER_PALLET
    baseline_travel = route["total_duration_sec"] - baseline_unload
    return max(baseline_travel, 0.0), baseline_unload


# ----------------------------- Simulation --------------------------------- #

def sample_demand(store, distributions, rng):
    mean, stdev = distributions[store]
    if stdev == 0:
        return max(mean, 0.0)
    return max(rng.gauss(mean, stdev), 0.0)  # demand can't be negative


def traffic_multiplier_from_sample(raw_sample, low, high):
    """
    Maps a raw Beta(alpha, beta) draw (in [0,1]) linearly onto [low, high]
    directly -- no artificial centering imposed on where the Beta shape's
    peak lands. The peak (mode) ends up wherever mode_raw naturally falls
    within [low, high]; nothing forces it to 1.0x. This is deliberately
    simpler than an earlier version that rescaled the two sides of the
    mode separately to force the peak onto 1.0x -- that centering was an
    assumption imposed on the data, not something the Beta shape itself
    justified.
    """
    return low + (high - low) * raw_sample


def run_simulation(routes, skipped, distributions, rng,
                    beta_alpha, beta_beta, low_multiplier, high_multiplier):
    """
    One simulated day under the fixed-loading model: pallet loads are
    locked at each route's planned baseline; only traffic affects cost.
    Demand is sampled and compared against the fixed load afterward, for
    the demand-met metric.
    """
    raw_sample = rng.betavariate(beta_alpha, beta_beta)
    traffic_multiplier = traffic_multiplier_from_sample(raw_sample, low_multiplier, high_multiplier)

    total_cost = 0.0
    total_requested = 0.0
    total_served = 0.0
    routes_with_shortfall = 0  # demand exceeded the PLANNED LOAD that day --
                                # NOT a capacity breach, see module docstring

    for route in routes:
        baseline_travel, _ = decompose_baseline(route)
        sim_demand = sum(sample_demand(s, distributions, rng) for s in route["stops"])
        sim_travel = baseline_travel * traffic_multiplier

        planned_load = route["total_pallets"]
        delivered = min(planned_load, sim_demand)
        unload_sec = planned_load * UNLOADING_SEC_PER_PALLET  # fixed, not demand-driven
        if sim_demand > planned_load:
            routes_with_shortfall += 1

        sim_duration = sim_travel + unload_sec
        cost = owned_cost(sim_duration) if route["mode"] == "owned" else leased_cost(sim_duration)

        total_cost += cost
        total_requested += sim_demand
        total_served += delivered

    for store in skipped:
        sim_demand = sample_demand(store, distributions, rng)
        total_cost += skip_cost(store)
        total_requested += sim_demand
        # served += 0 -- skipped stores are never served, by definition

    demand_met_pct = (total_served / total_requested * 100) if total_requested > 0 else 100.0
    return total_cost, demand_met_pct, routes_with_shortfall


def percentile(sorted_vals, p):
    """Linear-interpolation percentile, no numpy dependency."""
    k = (len(sorted_vals) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def confidence_interval(values, level=CI_LEVEL):
    s = sorted(values)
    tail = (1 - level) / 2
    return percentile(s, tail), percentile(s, 1 - tail)


def plot_results(costs, demand_met_pcts, shortfalls, summary, n_routes, output_dir, title=None):
    """Histograms of the simulated distributions with mean + CI marked --
    no comparison, just a visual read on each metric's spread. Skipped
    gracefully (with a note) if matplotlib isn't available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping CI plot)")
        return None

    panels = [
        (costs, "cost", "Total Cost ($)", lambda v: f"${v:,.0f}"),
        (demand_met_pcts, "demand_met_pct", "Demand Met (%)", lambda v: f"{v:.1f}%"),
        (shortfalls, "routes_with_shortfall", f"Routes w/ Shortfall (of {n_routes})", lambda v: f"{v:.1f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, (values, key, subtitle, fmt) in zip(axes, panels):
        mean, (lo, hi) = summary[key]
        ax.hist(values, bins=30, color="#2c5282", alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(mean, color="#c0392b", linestyle="--", linewidth=1.6, label=f"mean = {fmt(mean)}")
        ax.axvline(lo, color="#555", linestyle=":", linewidth=1.3)
        ax.axvline(hi, color="#555", linestyle=":", linewidth=1.3,
                    label=f"{int(CI_LEVEL*100)}% CI = [{fmt(lo)}, {fmt(hi)}]")
        ax.set_title(subtitle, fontsize=11.5, fontweight="bold")
        ax.set_ylabel("Simulations")
        ax.legend(fontsize=8.3, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    suptitle = title or f"Monte Carlo Results -- Fixed Loading (n={len(costs)} simulations)"
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
    parser.add_argument("--day", default=DAY)
    parser.add_argument("--n-sim", type=int, default=N_SIMULATIONS)
    parser.add_argument("--beta-alpha", type=float, default=BETA_ALPHA)
    parser.add_argument("--beta-beta", type=float, default=BETA_BETA)
    parser.add_argument("--low-multiplier", type=float, default=LOW_MULTIPLIER,
                         help="traffic multiplier at beta_sample=0 (best case)")
    parser.add_argument("--high-multiplier", type=float, default=HIGH_MULTIPLIER,
                         help="traffic multiplier at beta_sample=1 (worst case)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    distributions = fit_demand_distributions(args.demand_csv, args.day)
    routes, skipped = load_plan(args.solution_csv)

    missing = [s for r in routes for s in r["stops"] if s not in distributions]
    missing += [s for s in skipped if s not in distributions]
    if missing:
        raise ValueError(f"Stores in the plan but not in the demand CSV: {set(missing)}")

    print(f"Loaded plan: {len(routes)} routes, {len(skipped)} skipped stores")
    print(f"Fitted demand distributions for {len(distributions)} stores from {args.day}s "
          f"({args.demand_csv})")
    print(f"Running {args.n_sim} simulations "
          f"(traffic ~ Beta({args.beta_alpha}, {args.beta_beta}) "
          f"mapped onto [{args.low_multiplier}, {args.high_multiplier}])...")

    costs, demand_met_pcts, shortfalls = [], [], []
    for _ in range(args.n_sim):
        c, d, s = run_simulation(routes, skipped, distributions, rng,
                                  args.beta_alpha, args.beta_beta,
                                  args.low_multiplier, args.high_multiplier)
        costs.append(c)
        demand_met_pcts.append(d)
        shortfalls.append(s)

    cost_mean, cost_ci = statistics.mean(costs), confidence_interval(costs)
    demand_mean, demand_ci = statistics.mean(demand_met_pcts), confidence_interval(demand_met_pcts)
    shortfall_mean, shortfall_ci = statistics.mean(shortfalls), confidence_interval(shortfalls)

    print(f"\n--- Results over {args.n_sim} simulations ---")
    print(f"Total cost:            mean=${cost_mean:,.2f}  "
          f"{int(CI_LEVEL*100)}% CI=[${cost_ci[0]:,.2f}, ${cost_ci[1]:,.2f}]")
    print(f"Demand met:            mean={demand_mean:.2f}%  "
          f"{int(CI_LEVEL*100)}% CI=[{demand_ci[0]:.2f}%, {demand_ci[1]:.2f}%]")
    print(f"Routes w/ shortfall:   mean={shortfall_mean:.2f}/{len(routes)}  "
          f"{int(CI_LEVEL*100)}% CI=[{shortfall_ci[0]:.0f}, {shortfall_ci[1]:.0f}]")

    out_path = Path(args.output_dir) / "montecarlo_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["simulation", "total_cost", "demand_met_pct", "routes_with_shortfall"])
        for i, (c, d, s) in enumerate(zip(costs, demand_met_pcts, shortfalls), start=1):
            writer.writerow([i, round(c, 2), round(d, 4), s])
    print(f"\nPer-simulation results written to {out_path}")

    summary = {
        "cost": (cost_mean, cost_ci),
        "demand_met_pct": (demand_mean, demand_ci),
        "routes_with_shortfall": (shortfall_mean, shortfall_ci),
    }
    shed_label = "No Shedding" if "no_shed" in Path(args.solution_csv).stem else "Shedding Allowed"
    plot_title = f"{args.day} -- {shed_label} (n={args.n_sim})"
    plot_path = plot_results(costs, demand_met_pcts, shortfalls, summary, len(routes),
                              args.output_dir, title=plot_title)
    if plot_path:
        print(f"CI plot written to {plot_path}")


if __name__ == "__main__":
    main()
