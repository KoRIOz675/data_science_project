import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================
# Load the data
# =========================================
data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")

# ==========================================
# Intialize quantitative answers dictionary
# ==========================================
quantitative_answers = {}

# ==========================================
# Question 1
# ==========================================
print("\n--- Question 1 ---")
total_cities_initial = data1.shape[0]
quantitative_answers['q1a'] = total_cities_initial
print(f"From how many cities do the weather data is extracted [q1a]? {total_cities_initial}")

cities_missing_data = data1[data1.isnull().any(axis=1)].shape[0]
quantitative_answers['q1b'] = cities_missing_data
print(f"How many cities are affected by missing measurements [q1b]? {cities_missing_data}")

if cities_missing_data > 0:
    print("Deleting cities with missing data.")
    data1.dropna(inplace=True)
    print(f"Number of cities after dropping missing data: {data1.shape[0]}")
else:
    print("No cities with missing data to delete.")
print(f"The remaining cities will be considered in the following of our analysis.")


# ==========================================
# Question  2
# ==========================================
print("\n--- Question 2 ---")
results_q2 = {}

# Lowest Minimum Temperature
min_temp_val = data1['Minimum_temperature'].min()
city_min_temp = data1.loc[data1['Minimum_temperature'] == min_temp_val, 'City'].iloc[0]
results_q2['q2a_city'] = city_min_temp
results_q2['q2a_value'] = min_temp_val
quantitative_answers['q2a'] = f"{city_min_temp} ({min_temp_val}°C)"
print(f"Lowest minimum temperature [q2a]: {min_temp_val}°C in {city_min_temp}")

# Highest Minimum Temperature
max_min_temp_val = data1['Minimum_temperature'].max()
city_max_min_temp = data1.loc[data1['Minimum_temperature'] == max_min_temp_val, 'City'].iloc[0]
results_q2['q2b_city'] = city_max_min_temp
results_q2['q2b_value'] = max_min_temp_val
quantitative_answers['q2b'] = f"{city_max_min_temp} ({max_min_temp_val}°C)"
print(f"Highest minimum temperature [q2b]: {max_min_temp_val}°C in {city_max_min_temp}")

# Lowest Maximum Temperature
min_max_temp_val = data1['Maximum_temperature'].min()
city_min_max_temp = data1.loc[data1['Maximum_temperature'] == min_max_temp_val, 'City'].iloc[0]
results_q2['q2c_city'] = city_min_max_temp
results_q2['q2c_value'] = min_max_temp_val
quantitative_answers['q2c'] = f"{city_min_max_temp} ({min_max_temp_val}°C)"
print(f"Lowest maximum temperature [q2c]: {min_max_temp_val}°C in {city_min_max_temp}")

# Highest Maximum Temperature
max_max_temp_val = data1['Maximum_temperature'].max()
city_max_max_temp = data1.loc[data1['Maximum_temperature'] == max_max_temp_val, 'City'].iloc[0]
results_q2['q2d_city'] = city_max_max_temp
results_q2['q2d_value'] = max_max_temp_val
quantitative_answers['q2d'] = f"{city_max_max_temp} ({max_max_temp_val}°C)"
print(f"Highest maximum temperature [q2d]: {max_max_temp_val}°C in {city_max_max_temp}")

# Lowest Rainfall
min_rainfall_val = data1['Rainfall'].min()
city_min_rainfall = data1.loc[data1['Rainfall'] == min_rainfall_val, 'City'].iloc[0]
results_q2['q2e_city'] = city_min_rainfall
results_q2['q2e_value'] = min_rainfall_val
quantitative_answers['q2e'] = f"{city_min_rainfall} ({min_rainfall_val} mm)"
print(f"Lowest rainfall [q2e]: {min_rainfall_val} mm in {city_min_rainfall}")

# Highest Rainfall
max_rainfall_val = data1['Rainfall'].max()
city_max_rainfall = data1.loc[data1['Rainfall'] == max_rainfall_val, 'City'].iloc[0]
results_q2['q2f_city'] = city_max_rainfall
results_q2['q2f_value'] = max_rainfall_val
quantitative_answers['q2f'] = f"{city_max_rainfall} ({max_rainfall_val} mm)"
print(f"Highest rainfall [q2f]: {max_rainfall_val} mm in {city_max_rainfall}")

# Lowest Sunshine Duration
min_sunshine_val = data1['Sunshine_duration'].min()
city_min_sunshine = data1.loc[data1['Sunshine_duration'] == min_sunshine_val, 'City'].iloc[0]
results_q2['q2g_city'] = city_min_sunshine
results_q2['q2g_value'] = min_sunshine_val
quantitative_answers['q2g'] = f"{city_min_sunshine} ({min_sunshine_val} hours)"
print(f"Lowest sunshine duration [q2g]: {min_sunshine_val} hours in {city_min_sunshine}")

# Highest Sunshine Duration
max_sunshine_val = data1['Sunshine_duration'].max()
city_max_sunshine = data1.loc[data1['Sunshine_duration'] == max_sunshine_val, 'City'].iloc[0]
results_q2['q2h_city'] = city_max_sunshine
results_q2['q2h_value'] = max_sunshine_val
quantitative_answers['q2h'] = f"{city_max_sunshine} ({max_sunshine_val} hours)"
print(f"Highest sunshine duration [q2h]: {max_sunshine_val} hours in {city_max_sunshine}")


# ==========================================
# Question  3
# ==========================================
print("\n--- Question 3 ---\n")
var_min_temp = data1['Minimum_temperature'].var()
quantitative_answers['q3a'] = round(var_min_temp, 2)
print(f"Variance of minimum temperature [q3a]: {var_min_temp:.2f}")

var_max_temp = data1['Maximum_temperature'].var()
quantitative_answers['q3b'] = round(var_max_temp, 2)
print(f"Variance of maximum temperature [q3b]: {var_max_temp:.2f}")

var_rainfall = data1['Rainfall'].var()
quantitative_answers['q3c'] = round(var_rainfall, 2)
print(f"Variance of total rainfall [q3c]: {var_rainfall:.2f}")

var_sunshine = data1['Sunshine_duration'].var()
quantitative_answers['q3d'] = round(var_sunshine, 2)
print(f"Variance of sunshine duration [q3d]: {var_sunshine:.2f}")

variances = {
    'Minimum Temperature': var_min_temp,
    'Maximum Temperature': var_max_temp,
    'Rainfall': var_rainfall,
    'Sunshine Duration': var_sunshine
}

lowest_variance_var_name = min(variances, key=variances.get)
highest_variance_var_name = max(variances, key=variances.get)

# print(f"\nVariable with lowest variance: {lowest_variance_var_name} ({variances[lowest_variance_var_name]:.2f})")
# print(f"Variable with highest variance: {highest_variance_var_name} ({variances[highest_variance_var_name]:.2f})")

column_map = {
    'Minimum Temperature': 'Minimum_temperature',
    'Maximum Temperature': 'Maximum_temperature',
    'Rainfall': 'Rainfall',
    'Sunshine Duration': 'Sunshine_duration'
}
lowest_variance_col = column_map[lowest_variance_var_name]
highest_variance_col = column_map[highest_variance_var_name]

print(f"\nVariable with lowest variance: {lowest_variance_var_name} (Column: {lowest_variance_col})")
print(f"Variable with highest variance: {highest_variance_var_name} (Column: {highest_variance_col})")


# ==========================================
# Question  4
# ==========================================
print("\n--- Question 4 ---\n")
mean_low_var = data1[lowest_variance_col].mean()
median_low_var = data1[lowest_variance_col].median()
std_low_var = data1[lowest_variance_col].std()

quantitative_answers['q4a'] = round(mean_low_var, 2)
quantitative_answers['q4b'] = round(median_low_var, 2)
quantitative_answers['q4c'] = round(std_low_var, 2)

print(f"Analysis for variable with lowest variance: {lowest_variance_var_name}")
print(f"Mean [q4a]: {mean_low_var:.2f}")
print(f"Median [q4b]: {median_low_var:.2f}")
print(f"Standard Deviation [q4c]: {std_low_var:.2f}")

plt.figure(figsize=(8, 6))
sns.histplot(data1[lowest_variance_col], kde=True)
plt.title(f"Histogram of {lowest_variance_var_name} (Lowest Variance)")
plt.xlabel(lowest_variance_var_name)
plt.ylabel("Frequency")
plt.axvline(mean_low_var, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_low_var:.2f}')
plt.axvline(median_low_var, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_low_var:.2f}')
plt.legend()
plt.show()


# ==========================================
# Question  5
# ==========================================
print("\n--- Question 5 ---\n")
mean_high_var = data1[highest_variance_col].mean()
median_high_var = data1[highest_variance_col].median()
std_high_var = data1[highest_variance_col].std()

quantitative_answers['q5a'] = round(mean_high_var, 2)
quantitative_answers['q5b'] = round(median_high_var, 2)
quantitative_answers['q5c'] = round(std_high_var, 2)

print(f"Analysis for variable with highest variance: {highest_variance_var_name}")
print(f"Mean [q5a]: {mean_high_var:.2f}")
print(f"Median [q5b]: {median_high_var:.2f}")
print(f"Standard Deviation [q5c]: {std_high_var:.2f}")

plt.figure(figsize=(8, 6))
sns.histplot(data1[highest_variance_col], kde=True)
plt.title(f"Histogram of {highest_variance_var_name} (Highest Variance)")
plt.xlabel(highest_variance_var_name)
plt.ylabel("Frequency")
plt.axvline(mean_high_var, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_high_var:.2f}')
plt.axvline(median_high_var, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_high_var:.2f}')
plt.legend()
plt.show()


# ==========================================
# Question  6
# ==========================================
print("\n--- Question 6 ---\n")