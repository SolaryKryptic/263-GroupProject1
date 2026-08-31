"""
enumerate_routes.py

Enumerates feasible truck routes (16-pallet capacity, geographically adjacent
stops only) for each day of the week, finds the optimal visiting order for
each candidate, prices it under the project's cost model, and prunes any
multi-stop route dominated by running its stops as separate single trips.

Output: one CSV per day (routes_<Day>.csv).
"""

import itertools
import pandas as pd
import numpy as np

TRUCK_CAPACITY = 16
SCHEDULED_TIME_SEC = 3.5 * 3600     # policy target - NOT where overtime starts
OVERTIME_THRESHOLD_SEC = 4 * 3600   # brief: overtime cost applies beyond 4 hours
UNLOAD_SEC_PER_PALLET = 18 * 60
NORMAL_RATE_PER_HOUR = 220
OVERTIME_RATE_PER_HOUR = 310
MAX_STOPS = 5
NEIGHBOR_THRESHOLD_SEC = 15 * 60  # a store must be near an already-chosen stop to join a route

WAREHOUSE = 'Warehouse'
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

durations_df = pd.read_csv('FoodstuffsDurations2026.csv', index_col=0)
STORE_NAMES = list(durations_df.index)
NAME_TO_IDX = {s: i for i, s in enumerate(STORE_NAMES)}
DUR = durations_df.values
WH_IDX = NAME_TO_IDX[WAREHOUSE]

ayush = pd.read_csv('ayush_estimates.csv')


def route_cost(total_seconds):
    """Base rate charged continuously up to 4 hours. Only time BEYOND 4 hours
    is billed at the overtime rate, rounded UP to the next whole hour."""
    if total_seconds <= OVERTIME_THRESHOLD_SEC:
        return (total_seconds / 3600) * NORMAL_RATE_PER_HOUR
    normal_cost = (OVERTIME_THRESHOLD_SEC / 3600) * NORMAL_RATE_PER_HOUR
    excess_seconds = total_seconds - OVERTIME_THRESHOLD_SEC
    excess_hours_billed = int(np.ceil(excess_seconds / 3600))
    return normal_cost + excess_hours_billed * OVERTIME_RATE_PER_HOUR


def best_order(idx_stops):
    best_time = None
    best_seq = None
    for perm in itertools.permutations(idx_stops):
        t = DUR[WH_IDX, perm[0]]
        for a, b in zip(perm[:-1], perm[1:]):
            t += DUR[a, b]
        t += DUR[perm[-1], WH_IDX]
        if best_time is None or t < best_time:
            best_time = t
            best_seq = perm
    return best_seq, best_time


def enumerate_feasible_subsets(demand_dict):
    stores_sorted = sorted(demand_dict.items(), key=lambda kv: kv[1])
    n = len(stores_sorted)

    def is_neighbor_of_chosen(candidate_idx, chosen_idx):
        return any(DUR[c, candidate_idx] <= NEIGHBOR_THRESHOLD_SEC or
                   DUR[candidate_idx, c] <= NEIGHBOR_THRESHOLD_SEC for c in chosen_idx)

    def backtrack(start, chosen, chosen_idx, load):
        if chosen:
            yield tuple(chosen)
        if len(chosen) == MAX_STOPS:
            return
        for i in range(start, n):
            s, d = stores_sorted[i]
            if load + d > TRUCK_CAPACITY:
                continue
            s_idx = NAME_TO_IDX[s]
            if chosen and not is_neighbor_of_chosen(s_idx, chosen_idx):
                continue
            yield from backtrack(i + 1, chosen + [s], chosen_idx + [s_idx], load + d)

    yield from backtrack(0, [], [], 0)


def build_day_routes(day):
    demand = ayush.set_index('Supermarket')[day]
    demand_dict = {s: int(d) for s, d in demand.items() if d > 0}

    rows = []
    single_stop_cost = {}

    for subset in enumerate_feasible_subsets(demand_dict):
        idx_stops = tuple(NAME_TO_IDX[s] for s in subset)
        seq_idx, drive_time = best_order(idx_stops)
        seq = [STORE_NAMES[i] for i in seq_idx]
        pallets = sum(demand_dict[s] for s in subset)
        # Route duration = driving time + unloading time (18 min/pallet,
        # applied once per pallet delivered on this route -- was previously
        # missing entirely, which silently under-priced every route).
        unload_time = pallets * UNLOAD_SEC_PER_PALLET
        total_time = drive_time + unload_time
        cost = route_cost(total_time)

        if len(subset) == 1:
            single_stop_cost[subset[0]] = cost

        rows.append({
            'stops_tuple': subset,
            'stops': ';'.join(seq),
            'num_stops': len(subset),
            'total_pallets': pallets,
            'total_duration_sec': round(total_time, 2),
            'cost': round(cost, 2),
            'over_policy_35h': total_time > SCHEDULED_TIME_SEC,
            'overtime_4h': total_time > OVERTIME_THRESHOLD_SEC,
        })

    df = pd.DataFrame(rows)

    def dominated(row):
        if row['num_stops'] == 1:
            return False
        separate_cost = sum(single_stop_cost[s] for s in row['stops_tuple'])
        return row['cost'] > separate_cost

    df['dominated'] = df.apply(dominated, axis=1)
    kept = df[~df['dominated']].reset_index(drop=True)

    kept['total_duration_hms'] = pd.to_timedelta(kept['total_duration_sec'], unit='s').apply(
        lambda td: str(td).split(' ')[-1])

    prefix = day[:3]
    kept.insert(0, 'route_id', [f"{prefix}_{i+1:05d}" for i in range(len(kept))])
    kept = kept.drop(columns=['stops_tuple', 'dominated'])
    return kept, len(df) - len(kept)


def main():
    for day in DAYS:
        routes, n_pruned = build_day_routes(day)
        routes.to_csv(f'routes_{day}.csv', index=False)
        n_overtime = routes['overtime_4h'].sum()
        print(f"{day}: {len(routes)} routes kept, {n_pruned} dominated pruned, "
              f"{n_overtime} genuinely over 4h (max stops: {routes['num_stops'].max()})")


if __name__ == "__main__":
    main()