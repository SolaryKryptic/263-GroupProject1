# 263-GroupProject1: Foodstuffs NZ Truck Delivery Optimisation

## Problem Overview

55 stores require daily pallet deliveries, Monday–Saturday. Foodstuffs operates
20 owned trucks (16-pallet capacity, two 3.5 h shifts each → 40 owned trips/day),
with wet-leased trucks (Linfox) available at $1,400 per 2 h block and overtime on
owned trucks at $310/h (vs $220/h standard). A fuel-reduction proposal is evaluated:
shedding up to 20% of stores at $1,500 per Pak 'n Save / $800 per other store.

Two variants are delivered:
- **No shedding (baseline)**
- **Shedding allowed (fuel-reduction)**

## Pipeline

1. **Demand estimation**: mean + 0.5σ per store × weekday from 6 weeks of training
   data (weeks 7–8 as unseen test period).
2. **Route-pool generation**: randomised-greedy construction: next stop chosen with
   probability ∝ 1/travel_time^3.5, 12% per-step early-stopping probability, 10
   forced starts per store, 3,000 build attempts/day, 16-pallet capacity cap.
3. **MILP optimisation**: binary route /
   wet-lease / shed variables; owned $220/h (first 4 h) then $310/h, leased
   $1,400/2-h block, shed $1,500/$800; fleet ≤ 40 owned trips, shed ≤ 20% of stores.
4. **Route visualisation**: Folium maps following the real road network, colour-coded
   by chain, shed stores greyed with strikethrough.
5. **Monte Carlo simulation**: traffic multipliers from the fitted right-skewed Beta
   distributions (multiplier = low + (high−low)·β, one shared draw per iteration),
   demand sampled N(mean, √mean), trucks loaded at the planned estimate; shortfalls
   trigger wet-lease top-ups. 1,000 iterations/day (seed 42).

## Data Sources

- `FoodstuffsDemand2026.csv`: historical daily demand, 55 stores × 56 days
- `FoodstuffsDurations2026.csv`: OSM road-network travel-time matrix
- `FoodstuffsLocations.csv`: store locations & chain types
- Auckland Transport traffic counts (July 2012 – June 2026) — Beta multiplier fit