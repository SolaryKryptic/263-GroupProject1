import pandas as pd

# Convert a long-format estimate file (Supermarket, Weekday, Pallet Size)
# into the wide format compare_estimates.py expects (one column per weekday).
long_format = pd.read_csv('ubeen_raw/Pallet_Size.csv')
wide_format = long_format.pivot(index='Supermarket', columns='Weekday', values='Pallet Size')

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
wide_format = wide_format[day_order]
wide_format.to_csv('estimates/estimate_ubeen.csv')
