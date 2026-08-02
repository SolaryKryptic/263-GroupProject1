import pandas as pd
import numpy as np
from scipy.stats import norm

# --- Load and tidy data ---
df = pd.read_csv('FoodstuffsDemand2026.csv')
tidy = df.melt(id_vars='Supermarket', var_name='Date', value_name='Demand')
tidy['Date'] = pd.to_datetime(tidy['Date'], format='%d/%m/%Y')
tidy['DayOfWeek'] = tidy['Date'].dt.day_name()

# --- Correct known data-entry errors (extra zero typos) ---
tidy.loc[(tidy['Supermarket'] == "Pak 'n Save Henderson") &
         (tidy['Date'] == '2026-06-10'), 'Demand'] = 12
tidy.loc[(tidy['Supermarket'] == 'Four Square Ellerslie') &
         (tidy['Date'] == '2026-06-03'), 'Demand'] = 4

# --- Remove King's Birthday (1 June 2026, Monday) - stores closed, demand = 0 ---
tidy = tidy[tidy['Date'] != '2026-06-01']

# --- Keep only the TRAINING weeks (first 6 weeks, up to 14 June 2026) ---
# The last 2 weeks are held out as a test set (matching compare_estimates.py),
# so they must NOT be used to build the estimate, or the comparison is unfair.
train = tidy[tidy['Date'] <= '2026-06-14']

# --- Estimate pallets required per store per day-of-week ---
# 80% service level: estimate covers actual demand on ~80% of days.
z = norm.ppf(0.8)

stats = (train.groupby(['Supermarket', 'DayOfWeek'])['Demand']
               .agg(mean='mean', std='std')
               .reset_index())
stats['PalletEstimate'] = np.ceil(stats['mean'] + z * stats['std'].fillna(0)).astype(int)
stats.loc[stats['DayOfWeek'] == 'Sunday', 'PalletEstimate'] = 0

# --- Reshape into a Supermarket x DayOfWeek pallet estimate table ---
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pallet_table = stats.pivot(index='Supermarket', columns='DayOfWeek', values='PalletEstimate')
pallet_table = pallet_table[day_order]

print(pallet_table)
pallet_table.to_csv('pallet_estimates.csv')

# --- Also save in the format compare_estimates.py expects ---
import os
os.makedirs('estimates', exist_ok=True)
pallet_table.reset_index().to_csv('estimates/estimate_tomi.csv', index=False)