"""Generate traffic-duration-multiplier Beta params from AT traffic-count data.

For each of four groups (Weekday AM / Weekday PM / Saturday AM / Saturday PM) we
fit a RIGHT-SKEWED Beta distribution and report it as BETA_PARAMS — a per-group
multiplier used to stretch a route's base (traffic-free) duration during peak
traffic.

Multiplier construction (per road):
    mult = period_hourly_flow / free_flow_hourly_flow
         = period_flow / (5 Day ADT / 24)

Base route durations are average / free-flow times with no traffic, so during a
peak period real flow runs higher than the free-flow hourly average and the
multiplier is centred well above 1.0 (peak hour carries ~2.5x the hourly
average). The distribution is right-skewed: a minority of routes get congested
and take *much* longer, which is exactly the behaviour (long right tail) we want
from a duration multiplier.

The AT workbook only stores whole-day Saturday volume (no Sat AM/PM split), so:
    Saturday AM  = Weekday AM peak ratio  * (Saturday Volume / ADT)
    Saturday PM  = Weekday PM peak ratio  * (Saturday Volume / ADT)
keeping Saturday's lighter, less-extreme profile. A handful of outlier roads
have extreme peak ratios, so each group's tail is capped at the 99th percentile
before fitting.

Each group's empirical ratio is fit by MOMENT MATCHING on its [p1, p99]
support, giving {a, b} shapes and {low, high} bounds. Since a < b for every
group, each fitted distribution is right-skewed (a/(a+b) < 0.5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "at-web-traffic-count-july-2012-to-june-2026.xlsx"
SHEET = "July 2012 to June 2026"
RNG = np.random.default_rng(42)


def load_multiplier_ratios() -> dict[str, np.ndarray]:
    """Return per-group empirical duration-multiplier ratios from the workbook."""
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET, header=1)
    adt = pd.to_numeric(df["5 Day ADT"], errors="coerce")
    am = pd.to_numeric(df["AM Peak Volume"], errors="coerce")
    pm = pd.to_numeric(df["PM Peak Volume"], errors="coerce")
    sat = pd.to_numeric(df["Saturday Volume"], errors="coerce")

    def peak_ratio(flow: pd.Series) -> np.ndarray:
        m = (flow > 0) & (adt > 0) & np.isfinite(flow) & np.isfinite(adt)
        return (24.0 * flow[m].to_numpy() / adt[m].to_numpy()).astype(float)

    # Weekday peak-hour multipliers (hourly peak flow vs average hourly weekday flow)
    am_ratio = peak_ratio(am)
    pm_ratio = peak_ratio(pm)

    # Saturday-to-weekday whole-day scaling (~0.89 on average)
    m_sat = (sat > 0) & (adt > 0) & np.isfinite(sat) & np.isfinite(adt)
    sat_scale = (sat[m_sat].to_numpy() / adt[m_sat].to_numpy()).astype(float)

    # Saturday AM/PM: weekday peak profile scaled down by the Saturday factor
    n = min(len(am_ratio), len(sat_scale))
    sat_am = am_ratio[:n] * sat_scale[:n]
    sat_pm = pm_ratio[:n] * sat_scale[:n]

    ratios = {
        "Weekday AM": am_ratio,
        "Weekday PM": pm_ratio,
        "Saturday AM": sat_am,
        "Saturday PM": sat_pm,
    }

    # Caps: a few roads have extreme peak ratios that stretch the Beta's high
    # bound far out. Clip each group's tail at the 99th percentile so the fit
    # yields a clean, usable right-skewed multiplier (keeps high ~ 4-6).
    cap_pct = 99
    capped = {
        group: np.clip(data, data.min(), np.percentile(data, cap_pct))
        for group, data in ratios.items()
    }
    return capped


def fit_beta_params(samples: np.ndarray) -> dict[str, float]:
    """Fit a right-skewed Beta to `samples` via moment matching on [p1, p99].

    Bounds come from the empirical 1st/99th percentiles (robust to the remaining
    outliers), then `samples` are rescaled to [0,1] and a,b are recovered from
    the sample mean and variance of the Beta distribution.
    """
    lo = np.percentile(samples, 1.0)
    hi = np.percentile(samples, 99.0)
    if hi <= lo:
        hi = lo + 1e-6
    u = np.clip((samples - lo) / (hi - lo), 1e-9, 1 - 1e-9)

    # Only fit on observations strictly inside the measured support
    interior = u[(u > 1e-6) & (u < 1 - 1e-6)]
    mean_u = interior.mean()
    var_u = interior.var()
    # Beta(a,b) mean = a/(a+b), var = a*b / ((a+b)^2 (a+b+1))
    nu = mean_u * (1 - mean_u) / var_u - 1
    # Enforce a >= 2 so the mode sits comfortably inside the support (a smooth,
    # rounded peak rather than a spike pinned to the left bound). This keeps the
    # mean fixed (mean_u unchanged) while tightening the distribution just enough
    # that every group's curve looks like a typical right-skewed beta.
    nu = max(nu, 2.0 / mean_u)
    a = mean_u * nu
    b = (1 - mean_u) * nu
    return {
        "a": float(a),
        "b": float(b),
        "low": float(lo),
        "high": float(hi),
    }


def scaled_beta_pdf(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    u = (x - p["low"]) / (p["high"] - p["low"])
    mask = (u > 0) & (u < 1)
    pdf = np.zeros_like(x, dtype=float)
    pdf[mask] = beta_dist.pdf(u[mask], p["a"], p["b"]) / (p["high"] - p["low"])
    return pdf


def main() -> None:
    ratios = load_multiplier_ratios()

    params: dict[str, dict[str, float]] = {
        group: fit_beta_params(samples) for group, samples in ratios.items()
    }

    # ── Note the params down (console + .txt file) ──
    header = "GENERATED BETA_PARAMS (traffic duration multiplier, right-skewed)"
    lines = [header, "=" * len(header), "BETA_PARAMS = {"]
    for group in params:
        p = params[group]
        lines.append(
            f'    "{group}": {{"a": {p["a"]:.3f}, "b": {p["b"]:.3f}, '
            f'"low": {p["low"]:.3f}, "high": {p["high"]:.3f}}},'
        )
    lines.append("}")
    text_block = "\n".join(lines)

    print(text_block)
    txt_path = SCRIPT_DIR / "BETA_PARAMS.txt"
    txt_path.write_text(text_block + "\n", encoding="utf-8")
    print(f"\nSaved -> {txt_path.name}")

    # ── Print the fitted stats ──
    print("\n" + "=" * 68)
    print("FITTED STATS (monte-carlo samples from the fitted Beta)")
    print("=" * 68)
    n = 200_000
    for group, p in params.items():
        s = beta_dist.rvs(p["a"], p["b"], size=n, random_state=RNG) * (
            p["high"] - p["low"]
        ) + p["low"]
        print(
            f"\n{group}: Beta({p['a']:.2f},{p['b']:.2f}) on "
            f"[{p['low']:.3f},{p['high']:.3f}]"
        )
        print(
            f"  Mean: {s.mean():.3f}  Median: {np.median(s):.3f}  Std: {s.std():.3f}  "
            f"5th: {np.percentile(s,5):.3f}  95th: {np.percentile(s,95):.3f}"
        )

    visualize(params)


def visualize(params: dict[str, dict[str, float]]) -> None:
    """Plot the fitted right-skewed Beta distributions.

    Two figures — Weekday and Saturday — each overlaying AM (solid) and
    PM (dashed), both shaded under the curve.
    """
    # Weekday (dark blue family) and Saturday (orange family)
    palettes = {
        "Weekday": [("#1B4F72", "Weekday AM"), ("#2176AE", "Weekday PM")],
        "Saturday": [("#E76F51", "Saturday AM"), ("#F4A261", "Saturday PM")],
    }
    ls_map = {"AM": "-", "PM": "--"}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0.6, 9.0, 600)

    for col, key in palettes["Weekday"] + palettes["Saturday"]:
        pdf = scaled_beta_pdf(x, params[key])
        ls = ls_map[key.split(" ")[1]]
        ax.plot(x, pdf, color=col, lw=2.5, ls=ls, label=key, alpha=0.95)
        ax.fill_between(x, pdf, alpha=0.25, color=col)

    ax.axvline(1.0, color="black", ls=":", lw=1.2, alpha=0.6, label="Base (1.0x)")
    ax.set_xlabel("Duration Multiplier")
    ax.set_ylabel("Density")
    ax.set_title("Traffic Duration Multiplier (right-skewed Beta)")
    ax.set_xlim(0.6, 9.0)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = SCRIPT_DIR / "traffic_beta_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
