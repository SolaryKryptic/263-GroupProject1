# 263-GroupProject1

# Problem: 

We need a single "typical demand" number per store **for each weekday** (Monday–Saturday) to build fixed routes that work week after week without rescheduling. Estimates are computed independently per day, so a store gets its own mean + 0.5σ for Monday, a different one for Tuesday, and so on. But actual demand jumps around, some Mondays a store needs 3 pallets, others 7.
# What the data showed:
- 6 weeks of data per store for each weekday (e.g., for Monday → mean ≈ 4.2, std ≈ 1.8 pallets)
- Pure mean (4.2) → under-delivers on ~50% of Mondays (those above average)
- Mean + 1σ (6.0) → over-delivers on ~84% of Mondays, expensive fleet
- Mean + 0.5σ (5.1) → covers ~69% of actual Mondays, reasonable buffer
- The same logic applies to every weekday, each using its own mean and std

# Why not just use the maximum?

Using the peak Monday (e.g., 8 pallets) would mean running half-empty trucks most weeks, wasting ~$200/trip. The 0.5σ buffer adds ~1 pallet/store on average, which translates to ~2 extra trips/week fleet-wide. That's ~$1,400 per week extra fleet cost vs. saving ~$7,000/week in wet-lease/recourse when demand spikes.

# Why not use a percentile (e.g., 80th)?

Percentiles are noisier with only 6 data points per store-weekday. The normal-approximation (mean + kσ) is more stable and the 0.5σ point naturally balances the two error costs:
- Under-delivery → wet-lease at $233/pallet or shed at $800
- Over-delivery → extra truck time at $66/pallet
The 0.5σ point roughly equalizes the expected marginal cost of one more pallet in either direction.

# Practical outcome
- Fleet: 201 owned trips/week (under 40/day cap)
- Wet-leasing: $0 (buffer absorbs normal variation)
- Shedding: 32 stores/week (cheaper than serving them on marginal days)
- Test-week recourse: ~$25k/week vs ~$45k for pure-mean plan