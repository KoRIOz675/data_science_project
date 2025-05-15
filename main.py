import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data
data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")

# Question 1
print("\n--- Question 1 ---\n")
total_cities = data1.shape[0]
cities_missing_data = data1[data1.isnull().any(axis=1)].shape[0]
print(f"Total number of cities: {total_cities}")
print(f"Number of cities with missing data: {cities_missing_data}")
data1.dropna(inplace=True)
print(f"Number of cities after dropping missing data: {data1.shape[0]}")

# Question 2
print("\n--- Question 2 ---\n")
cols_q2 = ['Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration']
cols_q2_desc = [
    "lowest minimum temperature", "highest minimum temperature",
    "lowest maximum temperature", "highest maximum temperature",
    "lowest rainfall", "highest rainfall",
    "lowest sunshine duration", "highest sunshine duration"
]

idx = 0
for col in cols_q2:
    min_val = data1[col].min()
    city_min = data1.loc[data1[col] == min_val, 'City']
    print(f"{cols_q2_desc[idx]}: {min_val} in {city_min.values[0]}")
    idx += 1


    max_val = data1[col].max()
    city_max = data1.loc[data1[col] == max_val, 'City']
    print(f"{cols_q2_desc[idx]}: {max_val} in {city_max.values[0]}")
    idx += 1

print("\ncomment ?")

# Question 3
print("\n--- Question 3 ---\n")