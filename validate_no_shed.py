import sys
import pandas as pd
import numpy as np

print('Validating NO-SHED solutions...', flush=True)

# Load all solutions
solutions = {}
for wd in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
    try:
        sol = pd.read_csv(f"{wd}_solution_no_shed.csv")
        solutions[wd] = sol
        print(f'Loaded {wd}: {len(sol)} routes', flush=True)
    except:
        print(f'No solution for {wd}', flush=True)

# Load estimates
est = pd.read_csv('estimations/0.5ayush6week-estimated_demand.csv')
est = est.set_index('Supermarket')

# Load test data
demand = pd.read_csv('FoodstuffsDemand2026.csv')
long = demand.melt(id_vars='Supermarket', value_vars=demand.columns[1:], var_name='Date', value_name='Demand')
long['Date'] = pd.to_datetime(long['Date'], dayfirst=True)
long['DayOfWeek'] = long['Date'].dt.day_name()
test = long[long['Date'] > pd.Timestamp('2026-06-14')]

loc = pd.read_csv('FoodstuffsLocations.csv')[['Supermarket', 'Type']].set_index('Supermarket')['Type'].to_dict()

OVER_COST = 66
UNDER_COST = 210

# For each solution, build per-store estimate
for wd in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
    if wd not in solutions:
        continue
    sol = solutions[wd]
    
    # Expand to per-store estimates
    est_del = {}
    for _, row in sol.iterrows():
        stores = [s.strip() for s in row['Stores'].split(',')]
        for s in stores:
            if s in est.index:
                est_del[s] = est.loc[s, wd]
    
    test_wd = test[test['DayOfWeek'] == wd]
    total_recourse = 0
    for col in sorted(test_wd['Date'].unique()):
        day_test = test_wd[test_wd['Date'] == col].set_index('Supermarket')['Demand']
        est_series = pd.Series({s: est_del.get(s, 0) for s in day_test.index}, index=day_test.index)
        shortfall = (day_test - est_series).clip(lower=0)
        excess = (est_series - day_test).clip(lower=0)
        
        # No shedding in no-shed version, so all shortfall = wet-lease
        wet = shortfall * UNDER_COST
        recourse = wet
        cost = (excess * OVER_COST + recourse).sum()
        total_recourse += cost
        print(f'{col.strftime("%d/%m")} {wd}: ${cost:.0f}', flush=True)

    print(f'Total recourse for {wd}: ${sum([cost for col in sorted(test[test["DayOfWeek"]==wd]["Date"].unique()) for cost in []]):.0f}', flush=True)

print('Done.', flush=True)