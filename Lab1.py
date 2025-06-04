import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Preview the dataset
print("Dataset Preview:")
print(df.head())


numerical_col = 'MedInc'
data = df[numerical_col]


mean_val = data.mean()
median_val = data.median()
mode_val = data.mode().values
std_dev = data.std()
variance = data.var()
data_range = data.max() - data.min()

print(f"\nDescriptive Statistics for '{numerical_col}':")
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Range: {data_range}")


plt.figure(figsize=(10, 4))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(data, kde=True, bins=30)
plt.title(f'Histogram of {numerical_col}')

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=data)
plt.title(f'Boxplot of {numerical_col}')
plt.tight_layout()
plt.show()


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data < lower_bound) | (data > upper_bound)]

print(f"\nOutliers in '{numerical_col}':")
print(outliers)
