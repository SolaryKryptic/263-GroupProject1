import pandas as pd

"""
Method: Ubeen
Idea is to take upper quartile. But since data is very limited, only 6 per store/weekday
Combine data from same store/weekday. But since some locations are larger, normalize it by 
first converting data as percentage deviation from location mean.

Final pallet size is calculated by having group mean + upper quartile percentage times group mean

Assumptions: Demand distribution is similar across same store type and weekday
Ignore kings birthday.
75% is good enough, since this means only 25% of days will not maximize demand, but only portion of demand
is unmet that day. Since spread of percentage deviation is roughly uniform, means only about
0.25*0.25=6% of total volume is unmet at most. 
Realistically, when demand is met in excess, the food will likely carry over to the next day, reducing 
total unmet demand. 
"""


"""
Step 1
Clean data 
Melt + Remove outliers
Outlier = Multiplied by 10, and King's birthday
"""
#Open data file to read as df
df = pd.read_csv("FoodstuffsDemand2026.csv")

#Melt data, so that each row starts with supermarket name, date, and demand
long = df.melt(id_vars=["Supermarket"], var_name="Date", value_name="Demand")

#Convert string date into numerical date type
long["Date"] = pd.to_datetime(long["Date"],format="%d/%m/%Y")

#Convert dates into weekday names, and add new column called "Weekday"
long["Weekday"] = long["Date"].dt.day_name()

#Remove outliers. Noticed outliers are multiplied by 10, so divide by 10.
#Demand can't exceed 16 since this is max transport capacity
#But demand at Pak 'n Save appears to sometimes? So increased it to 20, since only 1 extreme outlier after
long["Demand"] = long["Demand"].astype(float)
long.loc[long["Demand"] > 20, "Demand"] /= 10

#Remove king's birthday data
long = long.loc[long["Date"] != "2026-06-01"]

#Use only first 6 weeks of data
start_date = "2026-05-04"
end_date = "2026-06-14"
long = long[long["Date"].between(start_date,end_date)]

"""
Step 2
Calculate mean of each locations per weekday
Calculate % deviation from mean and add as a new column
"""
#Calculate Average demand per store per weekday, and add new column called "Avg"
long["Avg"] = long.groupby(["Supermarket","Weekday"])["Demand"].transform("mean")

#Calculate % deviation from mean, and store as new column "% dev"
long["% dev"] = (long["Demand"] - long["Avg"])/long["Avg"]

"""
Step 3
Group % Deviation by store type and weekday
Calculate upper quartile for each weekday by store type
"""
#Create a function that determines store type
def store_type(name):
    if name.startswith("New World"):
        return "New World"
    elif name.startswith("Four Square"):
        return "Four Square"
    elif name.startswith("Pak 'n Save"):
        return "Pak 'n Save"
    else:
        return "Other"

#Create new column called "Type"
long["Type"] = long["Supermarket"].apply(store_type)

#Calculate upper quartile of % dev by Type and Weekday, add new column as "UQ"
long["UQ"] = long.groupby(["Type","Weekday"])["% dev"].transform(lambda x: x.quantile(0.75))

"""
Step 4
Using UQ, calculate pallet size
By adding UQ percentage to group mean
"""
#Calculate pallet size per store per weekday, store as new column called pallet size
#Round to nearest int
long["Pallet Size"] = (long["Avg"]+long["UQ"]*long["Avg"]).round()

#Order weekday
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
long["Weekday"] = pd.Categorical(long["Weekday"], categories=weekday_order, ordered=True)
long = long.sort_values(by=["Weekday"]).reset_index(drop=True)

#Create summary
summary_df = long.groupby(["Supermarket","Weekday"])["Pallet Size"].first().reset_index()

#Export
summary_df.to_csv("Pallet_Size.csv", index=False)














