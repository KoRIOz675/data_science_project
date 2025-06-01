import pandas as pd
import comments
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
# Initialize comments module
# ==========================================
comment_data_export = {}

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
comment_data_export['q2'] = results_q2


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
comment_data_export['q3_variances'] = {
    'var_min_temp': var_min_temp,
    'var_max_temp': var_max_temp,
    'var_rainfall': var_rainfall,
    'var_sunshine': var_sunshine
}
comment_data_export['q3_lowest_var_name'] = lowest_variance_var_name
comment_data_export['q3_highest_var_name'] = highest_variance_var_name


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
comment_data_export['q4_stats'] = {
    'var_name': lowest_variance_var_name,
    'variance_val': variances[lowest_variance_var_name],
    'mean_val': mean_low_var,
    'median_val': median_low_var,
    'std_val': std_low_var
}


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
comment_data_export['q5_stats'] = {
    'var_name': highest_variance_var_name,
    'variance_val': variances[highest_variance_var_name],
    'mean_val': mean_high_var,
    'median_val': median_high_var,
    'std_val': std_high_var
}


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

quantitative_answers['q6a'] = f"{most_pos_corr_pair[0]} & {most_pos_corr_pair[1]} ({most_pos_corr_val:.3f})"
quantitative_answers['q6b'] = f"{most_neg_corr_pair[0]} & {most_neg_corr_pair[1]} ({most_neg_corr_val:.3f})"
quantitative_answers['q6c'] = f"{least_corr_pair[0]} & {least_corr_pair[1]} ({least_corr_val:.3f})"

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

quantitative_answers['q8a'] = round(q8a_val, 2)
quantitative_answers['q8b'] = round(q8b_val, 2)

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
# Save quantitative answers and comments
# ==========================================
csv_filename = "answers_MAFILLE_MAILLARD.csv"
output_df_list = []
for q_id, ans_val in quantitative_answers.items():
    output_df_list.append({'QuestionID': q_id, 'Answer': ans_val})
answers_df = pd.DataFrame(output_df_list)
try:
    answers_df.to_csv(csv_filename, index=False)
    print(f"\nQuantitative answers saved to {csv_filename}")
except Exception as e:
    print(f"\nError saving quantitative answers to CSV: {e}")

comment_data_filename = "comment_data.json"
try:
    with open(comment_data_filename, 'w') as f:
        json.dump(comment_data_export, f, indent=4, allow_nan=True)
    print(f"Data for comments saved to {comment_data_filename}")
except TypeError as e:
    print(f"Error saving comment data to JSON: {e}. Some data might not be JSON serializable (e.g. numpy types not converted).")
except Exception as e:
    print(f"An unexpected error occurred while saving comment data: {e}")
