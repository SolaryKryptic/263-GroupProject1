"""
Monte Carlo simulation for Foodstuffs trucking routes.

For each day (Mon-Sat), runs N iterations of random demand + traffic
variation on the MILP solution, producing a distribution of actual costs.

MODEL:
  - Route plan is FIXED (from MILP solution: which routes, which stores)
  - Truck loading is FIXED (estimated pallets from the route CSV)
  - Route duration is precomputed in the CSV (travel + unloading for estimates)
  - Traffic variation: duration gets multiplied by a random Beta factor
  - Demand variation: actual store demand differs from estimates, creating
    shortfalls (stores don't get enough pallets). Does NOT affect duration
    or cost — the truck carries the planned load regardless.

Traffic model: Per-route Beta-distributed multiplier on total duration.
  - Weekday AM: Beta(3,7) scaled to [0, 3.33]  (heaviest tail)
  - Weekday PM: Beta(3.5,6.5) scaled to [0, 3.0]
  - Saturday AM: Beta(5,6) scaled to [0, 2.2]
  - Saturday PM: Beta(6,5.5) scaled to [0, 2.0]

Demand model: Normal(estimate, sqrt(estimate)), floored at 0.
  - Estimates from 0_5ayush6week-estimated_demand.csv (mean + 0.5 sigma).
  - If actual > estimate at a store → shortfall (under-served).
  - If actual <= estimate → no issue, store gets full estimated load.

Cost model (matches solve_lp3.py):
  - Owned: $220/hr first 4 hrs, $310/hr overtime (per-minute prorated)
  - Leased: $1400 per 2-hr block (ceiling)
  - Shed: $1500 PnS, $800 other
"""

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import beta as beta_dist

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
})

# ── CONFIG ──
N_ITER = 5000
OWNED_RATE = 220.0
OVERTIME_RATE = 310.0
OVERTIME_THRESHOLD_HR = 4.0
LEASED_RATE_PER_BLOCK = 1400.0
LEASED_BLOCK_HR = 2.0
SHED_COST_PNS = 1500.0
SHED_COST_OTHER = 800.0

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
SOL_DIR_SHED = "Route generation/Solutions/Shedding Allowed"
SOL_DIR_NOSHED = "Route generation/Solutions/Shedding not allowed"
DEMAND_CSV = "estimations/0_5ayush6week-estimated_demand.csv"

# ── BETA DISTRIBUTIONS (right-skewed) ──
BETA_PARAMS = {
    "Weekday AM": {"a": 3.0, "b": 7.0, "low": 0.0, "high": 3.33},
    "Weekday PM": {"a": 3.5, "b": 6.5, "low": 0.0, "high": 3.0},
    "Saturday AM": {"a": 5.0, "b": 6.0, "low": 0.0, "high": 2.2},
    "Saturday PM": {"a": 6.0, "b": 5.5, "low": 0.0, "high": 2.0},
}


# ── COST FUNCTIONS ──
def owned_cost(duration_sec):
    hours = duration_sec / 3600.0
    normal = min(hours, OVERTIME_THRESHOLD_HR)
    overtime = max(hours - OVERTIME_THRESHOLD_HR, 0.0)
    return OWNED_RATE * normal + OVERTIME_RATE * overtime


def leased_cost(duration_sec):
    hours = duration_sec / 3600.0
    return LEASED_RATE_PER_BLOCK * math.ceil(hours / LEASED_BLOCK_HR)


def skip_cost(store_name):
    return SHED_COST_PNS if "Pak 'n Save" in store_name else SHED_COST_OTHER


# ── LOADERS ──
def load_demand(path):
    """Returns {store_name: {day: pallet_count, ...}} and store types."""
    demand = {}
    types = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            store = row["Supermarket"]
            demand[store] = {}
            for day in DAYS:
                demand[store][day] = int(row[day])
            types[store] = "Pak" if "Pak 'n Save" in store else "Other"
    return demand, types


def load_solution(path):
    """Returns list of route dicts and list of skipped stores."""
    routes = []
    skipped = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["route_id"].startswith("SKIP_"):
                skipped.append(row["stops"])
            elif row["route_id"] == "TOTAL":
                continue
            else:
                routes.append({
                    "route_id": row["route_id"],
                    "mode": row["mode"],
                    "stops": row["stops"].split(";"),
                    "total_pallets": int(row["total_pallets"]),
                    "total_duration_sec": float(row["total_duration_sec"]),
                    "cost": float(row["cost"]),
                })
    return routes, skipped


# ── SIMULATION ──
def simulate_day(day, routes, skipped, demand, types, rng):
    """Run one Monte Carlo iteration for a single day.

    Duration = precomputed route duration x traffic multiplier.
    Demand variation only creates shortfalls (does not affect duration/cost).
    """
    total_owned = 0.0
    total_leased = 0.0
    total_shed = 0.0
    total_overtime_min = 0.0
    n_routes_with_shortfall = 0

    total_est_pallets = 0
    total_delivered_pallets = 0

    am_key = "Saturday AM" if day == "Saturday" else "Weekday AM"
    pm_key = "Saturday PM" if day == "Saturday" else "Weekday PM"
    mid = len(routes) // 2

    for i, r in enumerate(routes):
        # 1. Traffic multiplier (Beta distribution)
        key = am_key if i < mid else pm_key
        d = BETA_PARAMS[key]
        traffic_mult = beta_dist.rvs(d["a"], d["b"]) * (d["high"] - d["low"]) + d["low"]

        # 2. Duration = precomputed duration x traffic multiplier
        new_duration = r["total_duration_sec"] * traffic_mult

        # 3. Cost by mode (based on duration only, not demand)
        if r["mode"] == "owned":
            c = owned_cost(new_duration)
            total_owned += c
            if new_duration > OVERTIME_THRESHOLD_HR * 3600:
                total_overtime_min += (new_duration - OVERTIME_THRESHOLD_HR * 3600) / 60
        elif r["mode"] == "leased":
            total_leased += leased_cost(new_duration)

        # 4. Demand variation — only for shortfalls
        route_has_shortfall = False
        for store in r["stops"]:
            est = demand[store][day]
            total_est_pallets += est
            std = max(np.sqrt(est), 0.5)
            actual = max(0, int(round(rng.normal(est, std))))
            if actual > est:
                # Shortfall: store needed more than estimated
                total_delivered_pallets += est  # truck only carried estimate
                route_has_shortfall = True
            else:
                total_delivered_pallets += actual  # store got what it needed (<= estimate)

        if route_has_shortfall:
            n_routes_with_shortfall += 1

    # Skipped stores (fixed cost, no variation)
    for s in skipped:
        total_shed += skip_cost(s)

    demand_met_pct = (total_delivered_pallets / total_est_pallets * 100) if total_est_pallets > 0 else 100.0

    return {
        "owned": total_owned,
        "leased": total_leased,
        "shed": total_shed,
        "total": total_owned + total_leased + total_shed,
        "overtime_min": total_overtime_min,
        "routes_with_shortfall": n_routes_with_shortfall,
        "demand_met_pct": demand_met_pct,
    }


def run_simulation(day, routes, skipped, demand, types, n_iter=N_ITER):
    """Run full Monte Carlo for one day."""
    results = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_iter):
        results.append(simulate_day(day, routes, skipped, demand, types, rng))
    return results


# ── VISUALIZATION ──
def plot_results(all_results, scenario_label):
    """Generate plots for all 6 days."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx, day in enumerate(DAYS):
        ax = axes[idx // 3, idx % 3]
        res = all_results[day]
        totals = [r["total"] for r in res]
        base = np.mean([r["owned"] for r in res]) + np.mean([r["shed"] for r in res])

        ax.hist(totals, bins=50, color="#2176AE", alpha=0.7, edgecolor="white", density=True)
        ax.axvline(np.mean(totals), color="red", ls="--", lw=2, label=f"Mean: ${np.mean(totals):,.0f}")
        ax.axvline(np.median(totals), color="orange", ls="-", lw=1.5, label=f"Median: ${np.median(totals):,.0f}")
        ax.set_title(day, fontsize=12, fontweight="bold")
        ax.set_xlabel("Simulated Total Cost ($)")
        ax.set_ylabel("Density")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=7)

    fig.suptitle(f"Monte Carlo Cost Distribution ({scenario_label})\n{N_ITER} iterations per day", fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f"simulate_{scenario_label.replace(' ', '_').lower()}_histograms.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")

    # ── Summary bar chart ──
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(DAYS))
    width = 0.25

    means_owned = [np.mean([r["owned"] for r in all_results[d]]) for d in DAYS]
    means_leased = [np.mean([r["leased"] for r in all_results[d]]) for d in DAYS]
    means_shed = [np.mean([r["shed"] for r in all_results[d]]) for d in DAYS]

    bars1 = ax2.bar(x - width, means_owned, width, label="Owned Fleet", color="#2176AE")
    bars2 = ax2.bar(x, means_leased, width, label="Wet-Lease Recourse", color="#E76F51")
    bars3 = ax2.bar(x + width, means_shed, width, label="Shed Penalty", color="#F4A261")

    ax2.set_xlabel("Day")
    ax2.set_ylabel("Mean Simulated Cost ($)")
    ax2.set_title(f"Mean Cost Breakdown by Day ({scenario_label})")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d[:3] for d in DAYS])
    ax2.legend()
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))

    # Add total cost labels on top
    for i, d in enumerate(DAYS):
        total = means_owned[i] + means_leased[i] + means_shed[i]
        ax2.text(i, total + 200, f"${total:,.0f}", ha="center", fontsize=8, fontweight="bold")

    fig2.tight_layout()
    fname2 = f"simulate_{scenario_label.replace(' ', '_').lower()}_barchart.png"
    fig2.savefig(fname2, dpi=150, bbox_inches="tight")
    print(f"Saved {fname2}")

    return fig, fig2


def plot_comparison(shed_results, noshed_results):
    """Side-by-side comparison of shed vs no-shed."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(DAYS))
    width = 0.35

    shed_totals = [np.mean([r["total"] for r in shed_results[d]]) for d in DAYS]
    noshed_totals = [np.mean([r["total"] for r in noshed_results[d]]) for d in DAYS]

    bars1 = ax.bar(x - width/2, shed_totals, width, label="With Shedding (fuel reduction)", color="#2176AE")
    bars2 = ax.bar(x + width/2, noshed_totals, width, label="No Shedding (baseline)", color="#E76F51")

    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Simulated Cost ($)")
    ax.set_title("Fuel Reduction Proposal: Shed vs No-Shed (Monte Carlo)")
    ax.set_xticks(x)
    ax.set_xticklabels([d[:3] for d in DAYS])
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))

    # Savings annotation
    for i in range(len(DAYS)):
        saving = noshed_totals[i] - shed_totals[i]
        pct = saving / noshed_totals[i] * 100
        color = "green" if saving > 0 else "red"
        ax.text(i, max(shed_totals[i], noshed_totals[i]) + 500,
                f"Save ${saving:,.0f}\n({pct:.1f}%)", ha="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig("simulate_shed_vs_noshed.png", dpi=150, bbox_inches="tight")
    print("Saved simulate_shed_vs_noshed.png")


def plot_noshed_bars(all_results):
    """3 bar graphs for no-shedding scenario:
    1. Total cost (simulations on y, cost on x)
    2. Demand met % (simulations on y, % on x)
    3. Routes with shortfall (simulations on y, count on x)
    """
    colors = ["#1B4F72", "#2176AE", "#5DADE2", "#E76F51", "#F4A261", "#2A9D8F"]

    # Collect all data across days
    all_totals = []
    all_demands = []
    all_shortfalls = []
    for day in DAYS:
        res = all_results[day]
        all_totals.extend([r["total"] for r in res])
        all_demands.extend([r["demand_met_pct"] for r in res])
        all_shortfalls.extend([r["routes_with_shortfall"] for r in res])

    # Shared bin edges across all days
    cost_bins = np.linspace(min(all_totals) - 500, max(all_totals) + 500, 40)
    dem_bins = np.linspace(min(all_demands) - 0.5, 100.5, 30)
    max_sf = max(all_shortfalls) if max(all_shortfalls) > 0 else 1
    sf_bins = np.arange(-0.5, max_sf + 1.5, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Plot 1: Total Cost Histogram ──
    ax = axes[0]
    for idx, day in enumerate(DAYS):
        res = all_results[day]
        totals = [r["total"] for r in res]
        counts, _ = np.histogram(totals, bins=cost_bins)
        bin_centers = (cost_bins[:-1] + cost_bins[1:]) / 2
        ax.bar(bin_centers, counts, width=(cost_bins[1] - cost_bins[0]) * 0.85,
               alpha=0.5, color=colors[idx], label=day[:3], edgecolor="none")

    ax.set_xlabel("Total Cost ($)", fontsize=12)
    ax.set_ylabel("Number of Simulations", fontsize=12)
    ax.set_title("Total Cost Distribution\n(No Shedding)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Plot 2: Demand Met % Histogram ──
    ax = axes[1]
    for idx, day in enumerate(DAYS):
        res = all_results[day]
        demands = [r["demand_met_pct"] for r in res]
        counts, _ = np.histogram(demands, bins=dem_bins)
        bin_centers = (dem_bins[:-1] + dem_bins[1:]) / 2
        ax.bar(bin_centers, counts, width=(dem_bins[1] - dem_bins[0]) * 0.85,
               alpha=0.5, color=colors[idx], label=day[:3], edgecolor="none")

    ax.set_xlabel("Demand Met (%)", fontsize=12)
    ax.set_ylabel("Number of Simulations", fontsize=12)
    ax.set_title("Demand Fulfillment Distribution\n(No Shedding)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Plot 3: Routes with Shortfall Histogram ──
    ax = axes[2]
    bar_width = 0.12
    day_offsets = np.arange(len(DAYS)) - (len(DAYS) - 1) / 2 * bar_width

    for idx, day in enumerate(DAYS):
        res = all_results[day]
        shortfalls = [r["routes_with_shortfall"] for r in res]
        counts, _ = np.histogram(shortfalls, bins=sf_bins)
        bin_centers = (sf_bins[:-1] + sf_bins[1:]) / 2
        ax.bar(bin_centers + idx * bar_width, counts, width=bar_width * 0.9,
               alpha=0.7, color=colors[idx], label=day[:3], edgecolor="none")

    ax.set_xlabel("Routes with Shortfall (actual demand > estimate at a store)", fontsize=12)
    ax.set_ylabel("Number of Simulations", fontsize=12)
    ax.set_title("Route Shortfall Distribution\n(No Shedding)", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(0, max_sf + 1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"Monte Carlo Simulation Results — No Shedding ({N_ITER} iterations)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("simulate_noshed_3metrics.png", dpi=150, bbox_inches="tight")
    print("Saved simulate_noshed_3metrics.png")


def print_summary(all_results, label):
    """Print summary stats for one scenario."""
    print(f"\n{'='*70}")
    print(f"  MONTE CARLO SUMMARY: {label}  ({N_ITER} iterations)")
    print(f"{'='*70}")
    print(f"{'Day':<12} {'Mean':>10} {'Median':>10} {'Std':>10} {'P5':>10} {'P95':>10} {'OT min':>8} {'Shortfall':>10} {'DemMet%':>8}")
    print("-" * 96)
    weekly_total = 0
    for day in DAYS:
        res = all_results[day]
        totals = np.array([r["total"] for r in res])
        ot = np.mean([r["overtime_min"] for r in res])
        sf = np.mean([r["routes_with_shortfall"] for r in res])
        dm = np.mean([r["demand_met_pct"] for r in res])
        weekly_total += np.mean(totals)
        print(f"{day:<12} ${np.mean(totals):>8,.0f} ${np.median(totals):>8,.0f} "
              f"${np.std(totals):>8,.0f} ${np.percentile(totals, 5):>8,.0f} "
              f"${np.percentile(totals, 95):>8,.0f} {ot:>7.0f}m {sf:>8.1f} {dm:>7.1f}%")
    print("-" * 96)
    print(f"{'WEEKLY':<12} ${weekly_total:>8,.0f}")
    print()


# ── MAIN ──
def main():
    print("Loading demand estimates...")
    demand, types = load_demand(DEMAND_CSV)

    shed_results = {}
    noshed_results = {}

    for day in DAYS:
        print(f"\nSimulating {day}...")

        # Load solutions
        sol_shed, skip_shed = load_solution(f"{SOL_DIR_SHED}/{day.lower()}_solution.csv")
        sol_noshed, skip_noshed = load_solution(f"{SOL_DIR_NOSHED}/{day.lower()}_solution.csv")

        print(f"  Shedding allowed:   {len(sol_shed)} routes, {len(skip_shed)} skipped")
        print(f"  No shedding:        {len(sol_noshed)} routes, {len(skip_noshed)} skipped")

        shed_results[day] = run_simulation(day, sol_shed, skip_shed, demand, types)
        noshed_results[day] = run_simulation(day, sol_noshed, skip_noshed, demand, types)

    # Summaries
    print_summary(shed_results, "With Shedding (Fuel Reduction)")
    print_summary(noshed_results, "No Shedding (Baseline)")

    # Plots
    print("\nGenerating plots...")
    plot_results(shed_results, "With Shedding")
    plot_results(noshed_results, "No Shedding")
    plot_comparison(shed_results, noshed_results)
    plot_noshed_bars(noshed_results)

    # Final comparison
    shed_weekly = sum(np.mean([r["total"] for r in shed_results[d]]) for d in DAYS)
    noshed_weekly = sum(np.mean([r["total"] for r in noshed_results[d]]) for d in DAYS)
    saving = noshed_weekly - shed_weekly
    print(f"\n{'='*50}")
    print(f"  WEEKLY COMPARISON")
    print(f"{'='*50}")
    print(f"  With shedding:    ${shed_weekly:,.0f}")
    print(f"  Without shedding: ${noshed_weekly:,.0f}")
    print(f"  Weekly savings:   ${saving:,.0f} ({saving/noshed_weekly*100:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
