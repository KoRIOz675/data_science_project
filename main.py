import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================================
# Intormation
# ---
# Helper for the comments on the code: (link)
# https://drive.google.com/file/d/1EOXI7JJufuLSJb174WdrniIgMtnlHgvG/view?usp=sharing, https://drive.google.com/file/d/1QfUrx2zLy4rXt3FqHXCZ0Nsl0ZZ37bbP/view?usp=sharing, https://drive.google.com/file/d/1rlxzvuJzHV0vT6tLDhM6CYugQi3irFx0/view?usp=sharing, https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221zF851ZV13z3B-DjdD2EjWs4l8S_y0oYT%22%5D,%22action%22:%22open%22,%22userId%22:%22113384902893457590721%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing
# =========================================


# =========================================
# Load the data
# =========================================
data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")

# ==========================================
# Initialize quantitative answers dictionary
# ==========================================
quantitative_answers = {}

# ==========================================
# Initialize data variables
# ==========================================
data2_paris24_dict = {  # Renamed to avoid confusion with DataFrame later
    "Year": [],
    "Month": [],
    "Maximum_temperature": []
}

for i in range(data2["Year"].size):
    if data2["Year"][i] == 2024:
        data2_paris24_dict["Year"].append(2024)
        data2_paris24_dict["Month"].append(data2["Month"][i])
        data2_paris24_dict["Maximum_temperature"].append(data2["Maximum_temperature"][i])

data2_paris23_dict = { # Renamed and corrected
    "Year": [],
    "Month": [],
    "Maximum_temperature": []
}

for i in range(data2["Year"].size):
    if data2["Year"][i] == 2023:
        data2_paris23_dict["Year"].append(2023) # Corrected
        data2_paris23_dict["Month"].append(data2["Month"][i]) # Corrected
        data2_paris23_dict["Maximum_temperature"].append(data2["Maximum_temperature"][i]) # Corrected

data2_paris24 = pd.DataFrame(data2_paris24_dict)

data2_paris23 = pd.DataFrame(data2_paris23_dict)


# ==========================================
# Initialize variables for PCA
# ==========================================
numerical_features = ['Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration']
X_pca = data1[numerical_features]
cities_for_pca = data1['City']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_features)


# ==========================================
# Question 1
# ==========================================
print("\n--- Question 1 ---")
total_cities_initial = data1.shape[0]
quantitative_answers['q1a'] = [total_cities_initial, None, None]
print(f"From how many cities do the weather data is extracted [q1a]? {total_cities_initial}")

cities_missing_data = data1[data1.isnull().any(axis=1)].shape[0]
quantitative_answers['q1b'] = [cities_missing_data, None, None]
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
quantitative_answers['q2a'] = [city_min_temp, min_temp_val, None]
print(f"Lowest minimum temperature [q2a]: {min_temp_val}°C in {city_min_temp}")

# Highest Minimum Temperature
max_min_temp_val = data1['Minimum_temperature'].max()
city_max_min_temp = data1.loc[data1['Minimum_temperature'] == max_min_temp_val, 'City'].iloc[0]
results_q2['q2b_city'] = city_max_min_temp
results_q2['q2b_value'] = max_min_temp_val
quantitative_answers['q2b'] = [city_max_min_temp, max_min_temp_val, None]
print(f"Highest minimum temperature [q2b]: {max_min_temp_val}°C in {city_max_min_temp}")

# Lowest Maximum Temperature
min_max_temp_val = data1['Maximum_temperature'].min()
city_min_max_temp = data1.loc[data1['Maximum_temperature'] == min_max_temp_val, 'City'].iloc[0]
results_q2['q2c_city'] = city_min_max_temp
results_q2['q2c_value'] = min_max_temp_val
quantitative_answers['q2c'] = [city_min_max_temp, min_max_temp_val, None]
print(f"Lowest maximum temperature [q2c]: {min_max_temp_val}°C in {city_min_max_temp}")

# Highest Maximum Temperature
max_max_temp_val = data1['Maximum_temperature'].max()
city_max_max_temp = data1.loc[data1['Maximum_temperature'] == max_max_temp_val, 'City'].iloc[0]
results_q2['q2d_city'] = city_max_max_temp
results_q2['q2d_value'] = max_max_temp_val
quantitative_answers['q2d'] = [city_max_max_temp, max_max_temp_val, None]
print(f"Highest maximum temperature [q2d]: {max_max_temp_val}°C in {city_max_max_temp}")

# Lowest Rainfall
min_rainfall_val = data1['Rainfall'].min()
city_min_rainfall = data1.loc[data1['Rainfall'] == min_rainfall_val, 'City'].iloc[0]
results_q2['q2e_city'] = city_min_rainfall
results_q2['q2e_value'] = min_rainfall_val
quantitative_answers['q2e'] = [city_min_rainfall, min_rainfall_val, None]
print(f"Lowest rainfall [q2e]: {min_rainfall_val} mm in {city_min_rainfall}")

# Highest Rainfall
max_rainfall_val = data1['Rainfall'].max()
city_max_rainfall = data1.loc[data1['Rainfall'] == max_rainfall_val, 'City'].iloc[0]
results_q2['q2f_city'] = city_max_rainfall
results_q2['q2f_value'] = max_rainfall_val
quantitative_answers['q2f'] = [city_max_rainfall, max_rainfall_val, None]
print(f"Highest rainfall [q2f]: {max_rainfall_val} mm in {city_max_rainfall}")

# Lowest Sunshine Duration
min_sunshine_val = data1['Sunshine_duration'].min()
city_min_sunshine = data1.loc[data1['Sunshine_duration'] == min_sunshine_val, 'City'].iloc[0]
results_q2['q2g_city'] = city_min_sunshine
results_q2['q2g_value'] = min_sunshine_val
quantitative_answers['q2g'] = [city_min_sunshine, min_sunshine_val, None]
print(f"Lowest sunshine duration [q2g]: {min_sunshine_val} hours in {city_min_sunshine}")

# Highest Sunshine Duration
max_sunshine_val = data1['Sunshine_duration'].max()
city_max_sunshine = data1.loc[data1['Sunshine_duration'] == max_sunshine_val, 'City'].iloc[0]
results_q2['q2h_city'] = city_max_sunshine
results_q2['q2h_value'] = max_sunshine_val
quantitative_answers['q2h'] = [city_max_sunshine, max_sunshine_val, None]
print(f"Highest sunshine duration [q2h]: {max_sunshine_val} hours in {city_max_sunshine}")


# ==========================================
# Question  3
# ==========================================
print("\n--- Question 3 ---\n")
var_min_temp = data1['Minimum_temperature'].var()
quantitative_answers['q3a'] = [round(var_min_temp, 2), None, None]
print(f"Variance of minimum temperature [q3a]: {var_min_temp:.2f}")

var_max_temp = data1['Maximum_temperature'].var()
quantitative_answers['q3b'] = [round(var_max_temp, 2), None, None]
print(f"Variance of maximum temperature [q3b]: {var_max_temp:.2f}")

var_rainfall = data1['Rainfall'].var()
quantitative_answers['q3c'] = [round(var_rainfall, 2), None, None]
print(f"Variance of total rainfall [q3c]: {var_rainfall:.2f}")

var_sunshine = data1['Sunshine_duration'].var()
quantitative_answers['q3d'] = [round(var_sunshine, 2), None, None]
print(f"Variance of sunshine duration [q3d]: {var_sunshine:.2f}")

variances = {
    'Minimum Temperature': var_min_temp,
    'Maximum Temperature': var_max_temp,
    'Rainfall': var_rainfall,
    'Sunshine Duration': var_sunshine
}

lowest_variance_var_name = min(variances, key=variances.get)
highest_variance_var_name = max(variances, key=variances.get)

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

quantitative_answers['q4a'] = [round(mean_low_var, 2), None, None]
quantitative_answers['q4b'] = [round(median_low_var, 2), None, None]
quantitative_answers['q4c'] = [round(std_low_var, 2), None, None]

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

quantitative_answers['q5a'] = [round(mean_high_var, 2), None, None]
quantitative_answers['q5b'] = [round(median_high_var, 2), None, None]
quantitative_answers['q5c'] = [round(std_high_var, 2), None, None]

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
weather_vars = data1[['Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration']]
correlation_matrix = weather_vars.corr()

print("Correlation matrix between weather variables:")
print(correlation_matrix)
correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_val = correlation_matrix.iloc[i, j]
        correlations.append(((col1, col2), corr_val))
sorted_correlations = sorted(correlations, key=lambda item: item[1])

most_pos_corr_pair, most_pos_corr_val = sorted_correlations[-1]
most_neg_corr_pair, most_neg_corr_val = sorted_correlations[0]
sorted_by_abs_corr = sorted(correlations, key=lambda item: abs(item[1]))
least_corr_pair, least_corr_val = sorted_by_abs_corr[0]

quantitative_answers['q6a'] = [most_pos_corr_pair[0], most_pos_corr_pair[1], round(most_pos_corr_val, 2)]
quantitative_answers['q6b'] = [most_neg_corr_pair[0], most_neg_corr_pair[1], round(most_neg_corr_val, 2)]
quantitative_answers['q6c'] = [least_corr_pair[0], least_corr_pair[1], round(least_corr_val, 2)]
print(f"\nMost positively correlated variables [q6a]: {most_pos_corr_pair[0]} and {most_pos_corr_pair[1]} with correlation {most_pos_corr_val:.3f}")
print(f"Most negatively correlated variables [q6b]: {most_neg_corr_pair[0]} and {most_neg_corr_pair[1]} with correlation {most_neg_corr_val:.3f}")
print(f"Least correlated variables (closest to 0) [q6c]: {least_corr_pair[0]} and {least_corr_pair[1]} with correlation {least_corr_val:.3f}")

def plot_correlation_with_cities(df, var1, var2, title):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x=var1, y=var2)
    for i in range(df.shape[0]):
        plt.text(df[var1].iloc[i], df[var2].iloc[i], df['City'].iloc[i], fontsize=8, ha='right')
    plt.title(title + f"\nCorrelation: {df[[var1, var2]].corr().iloc[0,1]:.3f}")
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.grid(True)
    plt.show()

plot_correlation_with_cities(data1, most_pos_corr_pair[0], most_pos_corr_pair[1], "Most Positively Correlated Variables")
plot_correlation_with_cities(data1, most_neg_corr_pair[0], most_neg_corr_pair[1], "Most Negatively Correlated Variables")
plot_correlation_with_cities(data1, least_corr_pair[0], least_corr_pair[1], "Least Correlated Variables")


# ==========================================
# Question  7
# ==========================================
print("\n--- Question 7 ---\n")
data1_for_city_corr = data1.set_index('City')[['Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration']]
city_correlation_matrix = data1_for_city_corr.T.corr()

print("C.F. heatmap of correlation matrix between cities")

plt.figure(figsize=(16, 12))
sns.heatmap(city_correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Between Cities (based on weather variables)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ==========================================
# Question  8
# ==========================================
print("\n--- Question 8 ---\n")
pca = PCA(n_components=2)
X_pca_transformed = pca.fit_transform(X_scaled_df)

pca_df = pd.DataFrame(data=X_pca_transformed, columns=['PC1', 'PC2'])
pca_df['City'] = cities_for_pca.values

explained_variance_ratio = pca.explained_variance_ratio_
q8a_val = explained_variance_ratio[0] * 100
q8b_val = explained_variance_ratio[1] * 100

quantitative_answers['q8a'] = [round(q8a_val, 2), None, None]
quantitative_answers['q8b'] = [round(q8b_val, 2), None, None]

print(f"Percentage of variance explained by PC1 [q8a]: {q8a_val:.2f}%")
print(f"Percentage of variance explained by PC2 [q8b]: {q8b_val:.2f}%")
print(f"Total variance explained by first two PCs: {q8a_val + q8b_val:.2f}%")

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
for i in range(pca_df.shape[0]):
    plt.text(pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i], pca_df['City'].iloc[i], fontsize=9, ha='left', va='bottom')
plt.xlabel(f"PC1 ({q8a_val:.2f}%)")
plt.ylabel(f"PC2 ({q8b_val:.2f}%)")
plt.title("PCA of Weather Data - First Two Principal Components")
plt.grid(True)
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.show()


# ==========================================
# Question  9
# ==========================================
print("\n--- Question 9 ---\n")
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=numerical_features)
print("Loadings (correlation of variables with PCs):")
print(loadings_df)

fig, ax = plt.subplots(figsize=(8, 8))
for i, var_name in enumerate(numerical_features):
    ax.arrow(0, 0, loadings_df.PC1[i], loadings_df.PC2[i], head_width=0.05, head_length=0.1, fc='r', ec='r')
    ax.text(loadings_df.PC1[i] * 1.15, loadings_df.PC2[i] * 1.15, var_name, color='r', ha='center', va='center')

circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
ax.add_artist(circle)

ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_xlabel(f"PC1 ({q8a_val:.2f}%)")
ax.set_ylabel(f"PC2 ({q8b_val:.2f}%)")
ax.set_title("Correlation Circle (Variables Factor Map)")
ax.grid(True)
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# ==========================================
# Question 10
# ==========================================
print("\n--- Question 10 ---\n")
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax, s=50, legend=False, hue=pca_df.index)
for i in range(pca_df.shape[0]):
    ax.text(pca_df['PC1'].iloc[i] * 1.02, pca_df['PC2'].iloc[i] * 1.02, pca_df['City'].iloc[i], fontsize=9, ha='left', va='bottom')
ax2= ax.twinx().twiny()

pc1_range = pca_df['PC1'].max() - pca_df['PC1'].min()
pc2_range = pca_df['PC2'].max() - pca_df['PC2'].min()
loading_pc1_max_abs = np.abs(loadings_df.PC1).max()
loading_pc2_max_abs = np.abs(loadings_df.PC2).max()

for i, var_name in enumerate(numerical_features):
    ax2.arrow(0, 0, loadings_df.PC1[i], loadings_df.PC2[i], fc='red', ec='red', length_includes_head=True)
    ax2.text(loadings_df.PC1[i] * 1.2, loadings_df.PC2[i] * 1.2, var_name, color='red', ha='center', va='center', fontsize=12, fontweight='bold')

ax.set_title("Biplot: PCA of Weather Data (Cities and Variables)")
ax.grid(True, linestyle='--', alpha=0.7)
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(ax.get_ylim())
ax2.set_xticks([])
ax2.set_yticks([])

plt.tight_layout()
plt.show()


# ==========================================
# Question 11
# ==========================================
print("\n--- Question 11 ---\n")

plt.figure(figsize=(10,8))
plt.plot(data2_paris24['Month'], data2_paris24['Maximum_temperature'], marker='o', linestyle='-')
plt.title("Evolution of maximum temperature in Paris in 2024.")
plt.xlabel("Month")
plt.ylabel('Maximum Temperature')
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# Question 12
# ==========================================
print("\n-- Question 12 --\n")
data2_paris_2024 = data2_paris24.copy()
data2_paris_2024['month_ID'] = range(len(data2_paris_2024))

best_n = -1
best_adj_r2 = -float('inf')
best_model_results = None
best_b0 = np.nan
best_b1 = np.nan
is_adj_r2_used = False

for n in range(1, 13):
    subset_data = data2_paris_2024.tail(n)

    if len(subset_data) < 2:
        continue

    X_reg = subset_data['month_ID']
    X_reg = sm.add_constant(X_reg)
    y_reg = subset_data['Maximum_temperature']

    model = sm.OLS(y_reg, X_reg).fit()

    current_r2_value = model.rsquared_adj
    current_is_adj = True

    if current_r2_value > best_adj_r2:
        best_adj_r2 = current_r2_value
        best_n = n
        best_model_results = model
        best_b0 = model.params['const']
        best_b1 = model.params['month_ID']
        is_adj_r2_used = current_is_adj


quantitative_answers['q12a'] = [best_n, None, None]
quantitative_answers['q12b'] = [round(best_adj_r2, 3), None, None]
quantitative_answers['q12c'] = [round(best_model_results.rsquared, 3), None, None]


quantitative_answers['q12d'] = [round(best_b0, 3), None, None]
quantitative_answers['q12e'] = [round(best_b1, 3), None, None]

print(f"\nOptimal value of n [q12a]: {best_n}")
print(f"Value of the associated coefficient of determination R2 - adjusted [q12b] or not [q12c]?")
print(f"  - The selection was based on: {'adjusted R2' if is_adj_r2_used else 'R2'}")
print(f"  - Value of (Adjusted) R2 [q12b]: {best_adj_r2:.3f}")
if not is_adj_r2_used:
    print(f"  - Value of R2 (not adjusted) [q12c_value]: {best_model_results.rsquared:.3f}")

print(f"Values for the optimal model (n={best_n}):")
print(f"  β0 (intercept) [q12d]: {best_b0:.3f}")
print(f"  β1 (slope for month_ID) [q12e]: {best_b1:.3f}")

print("\nQuantitative and Visual Analysis of the Optimal Model:")
print(best_model_results.summary())


best_n_subset_data = data2_paris_2024.tail(best_n)
plt.figure(figsize=(10, 6))
plt.scatter(best_n_subset_data['month_ID'], best_n_subset_data['Maximum_temperature'], label='Actual Data (last n months)')
pred_y = best_model_results.predict(sm.add_constant(best_n_subset_data['month_ID']))
plt.plot(best_n_subset_data['month_ID'], pred_y, color='red', label=f'Linear Regression (n={best_n}, Adj R2={best_adj_r2:.3f})')
plt.xlabel("month_ID (0=Jan, ..., 11=Dec)")
plt.ylabel("Maximum Temperature (°C)")
plt.title(f"Optimal Simple Linear Regression Model for Paris 2024 Max Temp (last n={best_n} months)")
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# Question 13
# ==========================================
print("\n-- Question 13 --\n")
jan_2025_month_id = 12
predicted_temp_jan_2025 = best_b0 + best_b1 * jan_2025_month_id

actual_temp_jan_2025 = 7.5
difference_pred_actual = predicted_temp_jan_2025 - actual_temp_jan_2025

quantitative_answers['q13a'] = [round(predicted_temp_jan_2025, 2), None, None]
quantitative_answers['q13b'] = [round(difference_pred_actual, 2), None, None]

print(f"Predicted temperature for January 2025 [q13a]: {predicted_temp_jan_2025:.2f} °C")
print(f"Actual temperature for January 2025: {actual_temp_jan_2025}°C")
print(f"Difference (Predicted - Actual) [q13b]: {difference_pred_actual:.2f} °C")


# ==========================================
# Question 14
# ==========================================
print("\n-- Question 14 --\n")
p_value_beta1 = best_model_results.pvalues['month_ID']
alpha = 0.05
is_significant = p_value_beta1 < alpha

quantitative_answers['q14a'] = [round(p_value_beta1, 4), None, None]
quantitative_answers['q14b'] = ["Yes" if is_significant else "No", None, None]

print(f"Null hypothesis: β1 = 0 (no linear relationship between Max Temp and month_ID for the chosen n months).")
print(f"Alternative hypothesis: β1 ≠ 0.")
print(f"P-value for β1 [q14a]: {p_value_beta1:.4f}")
print(f"Significance level α: {alpha}")
if is_significant:
    print(f"Since p-value ({p_value_beta1:.4f}) < alpha ({alpha}), we reject the null hypothesis. [q14b]")
    print("Conclusion [q14b]: Yes, there is a statistically significant linear relationship between maximum temperature and month_ID for the selected n-month period at the 5% significance level.")
else:
    print(f"Since p-value ({p_value_beta1:.4f}) >= alpha ({alpha}), we fail to reject the null hypothesis. [q14b]")
    print("Conclusion [q14b]: No, there is not a statistically significant linear relationship between maximum temperature and month_ID for the selected n-month period at the 5% significance level.")


# ==========================================
# Question 15
# ==========================================
print("\n-- Question 15 --\n")
plt.figure(figsize=(12,7))
plt.plot(data2_paris23['Month'], data2_paris23['Maximum_temperature'], marker='o', linestyle='-', label='Paris\' Temperature in 2023')
plt.plot(data2_paris24['Month'], data2_paris24['Maximum_temperature'], marker='o', linestyle='--', label='Paris\' Temperature in 2024')
plt.title("Evolution of temperature in Paris in 2023 and 2024")
plt.xlabel('Month')
plt.ylabel('Maximum Temperature (°C)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

all_paris_temps = pd.concat([
    data2_paris23[['Year', 'Month', 'Maximum_temperature']],
    data2_paris24[['Year', 'Month', 'Maximum_temperature']]
]).sort_values(['Year', 'Month']).reset_index(drop=True)
df_multi = all_paris_temps.copy()
for lag in range(1, 13):
    df_multi[f'T_lag_{lag}'] = df_multi['Maximum_temperature'].shift(lag)
train_data_multi = df_multi[df_multi['Year'] == 2024].copy()

y_multi_train = train_data_multi['Maximum_temperature']
X_multi_train_full = train_data_multi[[f'T_lag_{lag}' for lag in range(1, 13)]]


print(f"Shape of X_multi_train_full: {X_multi_train_full.shape}")
print(f"Shape of y_multi_train: {y_multi_train.shape}")

# ==========================================
# Question 16
# ==========================================
print("\n-- Question 16 --\n")
num_predictor_vars = X_multi_train_full.shape[1]
total_combinations = 0
for k in range(1, num_predictor_vars + 1):
    from math import comb
    total_combinations += comb(num_predictor_vars, k)

quantitative_answers['q16a'] = [total_combinations, None, None]
print(f"Total number of possible combinations of variables (1 to {num_predictor_vars} predictors) [q16a]:{total_combinations}")

best_adj_r2_multi = -float('inf')
best_combo_multi = None
best_model_multi = None
num_selected_vars_multi = 0

max_k_predictors = X_multi_train_full.shape[0] - 2

all_feature_names = list(X_multi_train_full.columns)
iter_count = 0
for k_features in range(1, num_predictor_vars + 1):
    
    for combo in itertools.combinations(all_feature_names, k_features):
        iter_count += 1
        X_subset = X_multi_train_full[list(combo)]
        X_subset_const = sm.add_constant(X_subset)

        if X_subset_const.shape[1] > 1:
            try:
                cond_num = np.linalg.cond(X_subset_const.astype(float).values)
                if cond_num > 1000 :
                    continue
            except np.linalg.LinAlgError:
                continue
        
        model_multi = sm.OLS(y_multi_train, X_subset_const).fit()
        
        if model_multi.df_resid < 1 :
            current_adj_r2 = -float('inf')
        else:
            current_adj_r2 = model_multi.rsquared_adj

        if current_adj_r2 > best_adj_r2_multi:
            best_adj_r2_multi = current_adj_r2
            best_combo_multi = combo
            best_model_multi = model_multi
            num_selected_vars_multi = k_features


if best_model_multi is None:
    print("Error: No suitable multivariate model found. This might happen if all combinations lead to issues.")
else:
    quantitative_answers['q16b'] = [num_selected_vars_multi, None, None]
    param_strings = []
    for var_name, param_val in best_model_multi.params.items():
        param_strings.append(f"{var_name}: {param_val:.3f}")

    print(f"\nCombination of variables that maximises the adjusted R2 coefficient:")
    print(f"  Number of selected variables [q16b_num_vars]: {num_selected_vars_multi}")
    print(f"  Selected variables [q16b_selected_vars]: {best_combo_multi}")
    print(f"  Optimal adjusted R2 [q16b_adj_r2]: {best_adj_r2_multi:.3f}")
    print(f"  Associated parameters [q16b_parameters]:")
    print(best_model_multi.params)

    print("\nFull summary for the best multivariate model:")
    print(best_model_multi.summary())
    f_pvalue_multi = best_model_multi.f_pvalue
    alpha_multi = 0.05
    is_significant_multi = f_pvalue_multi < alpha_multi

    print(f"\nOverall model significance (F-test):")
    print(f"  F-statistic p-value: {f_pvalue_multi:.4f}")
    if is_significant_multi:
        print(f"  Since p-value ({f_pvalue_multi:.4f}) < alpha ({alpha_multi}), we reject the null hypothesis that all coefficients (excluding intercept) are zero. [q16_overall_significant]")
        print(f"  Conclusion: Yes, there is a statistically significant linear relationship overall between the selected lagged temperatures and the current month's temperature at the 5% significance level.")
    else:
        print(f"  Since p-value ({f_pvalue_multi:.4f}) >= alpha ({alpha_multi}), we fail to reject the null hypothesis. [q16_overall_significant]")
        print(f"  Conclusion: No, there is not a statistically significant linear relationship overall.")


# ==========================================
# Question 17
# ==========================================
print("\n-- Question 17 --\n")
jan_2025_predictors_raw = {}
temps_2024_list = data2_paris24['Maximum_temperature'].tolist()

for lag_num in range(1, 13):
    jan_2025_predictors_raw[f'T_lag_{lag_num}'] = temps_2024_list[12 - lag_num]

jan_2025_predictors_df = pd.DataFrame([jan_2025_predictors_raw])
jan_2025_selected_predictors = jan_2025_predictors_df[list(best_combo_multi)]
jan_2025_selected_predictors_const = sm.add_constant(jan_2025_selected_predictors, has_constant='add')
jan_2025_selected_predictors_const = jan_2025_selected_predictors_const[best_model_multi.model.exog_names]


predicted_temp_jan_2025_multi = best_model_multi.predict(jan_2025_selected_predictors_const)[0]
actual_temp_jan_2025_q17 = 7.5
difference_q17 = predicted_temp_jan_2025_multi - actual_temp_jan_2025_q17

quantitative_answers['q17'] = [round(difference_q17, 2), None, None]
print(f"Predicted temperature for January 2025 (multivariate model): {predicted_temp_jan_2025_multi:.2f} °C")
print(f"Actual temperature for January 2025: {actual_temp_jan_2025_q17}°C")
print(f"Difference (Predicted - Actual) for Jan 2025 [q17]: {difference_q17:.2f} °C")


# ==========================================
# Save quantitative answers and comments
# ==========================================
csv_filename = "answers_MAFILLE_MAILLARD_MERAD.csv"
output_df_list = []
for q_id, ans_vals in quantitative_answers.items():
    output_df_list.append({'Question': q_id, 'Answer1': ans_vals[0], 'Answer2': ans_vals[1], 'Answer3': ans_vals[2]})
answers_df = pd.DataFrame(output_df_list)

try:
    answers_df.to_csv(csv_filename, index=False)
    print(f"\nQuantitative answers saved to {csv_filename}")
except Exception as e:
    print(f"\nError saving quantitative answers to CSV: {e}")
