import sys
import pandas as pd
import numpy as np

print('hello', flush=True)

sol = pd.read_csv('monday_solution.csv')
est = pd.read_csv('estimations/0.5ayush6week-estimated_demand.csv')
est = est.set_index('Supermarket')
monday_est = est['Monday'].astype(int)

expanded = []
for _, row in sol.iterrows():
    stores = [s.strip() for s in row['Stores'].split(',')]
    for s in stores:
        if s in monday_est.index:
            expanded.append(s)

print('expanded:', len(expanded), flush=True)

demand = pd.read_csv('FoodstuffsDemand2026.csv')
long = demand.melt(id_vars='Supermarket', value_vars=demand.columns[1:], var_name='Date', value_name='Demand')
long['Date'] = pd.to_datetime(long['Date'], dayfirst=True)
long['DayOfWeek'] = long['Date'].dt.day_name()
test = long[(long['Date'] > pd.Timestamp('2026-06-14')) & (long['DayOfWeek'] == 'Monday')]

loc = pd.read_csv('FoodstuffsLocations.csv')[['Supermarket', 'Type']].set_index('Supermarket')['Type'].to_dict()

OVER_COST = 66
UNDER_COST = 210

total = 0
for col in sorted(test['Date'].unique()):
    day_test = test[test['Date'] == col].set_index('Supermarket')['Demand']
    est_del = pd.Series({s: monday_est[s] for s in expanded if s in day_test.index}, index=day_test.index).fillna(0)
    shortfall = (day_test - est_del).clip(lower=0)
    excess = (est_del - day_test).clip(lower=0)
    
    shed_c = np.array([1500 if loc.get(s, 'Other') == "Pak 'n Save" else 800 for s in day_test.index]) * (shortfall > 0)
    wet = shortfall * UNDER_COST
    use_shed = (shed_c < wet) & (shortfall > 0)
    recourse = np.where(use_shed, shed_c, shortfall * UNDER_COST)
    cost = (excess * OVER_COST + recourse).sum()
    total += cost
    print(f'{col.strftime("%d/%m")}: ${cost:.0f}', flush=True)

print(f'Total: ${total:.0f}', flush=True)