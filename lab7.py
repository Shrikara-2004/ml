import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\Boston housing dataset - Boston housing dataset.csv")

# Fill missing values (without loop)
df['CRIM'].fillna(df['CRIM'].median(), inplace=True)
df['ZN'].fillna(df['ZN'].median(), inplace=True)
df['INDUS'].fillna(df['INDUS'].median(), inplace=True)
df['AGE'].fillna(df['AGE'].median(), inplace=True)
df['LSTAT'].fillna(df['LSTAT'].median(), inplace=True)
df['CHAS'].fillna(df['CHAS'].mode()[0], inplace=True)


# Histogram & Boxplot 
for feature in df.columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df[feature].hist(bins=20, edgecolor='black', color='skyblue')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df[feature], vert=False)
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()

# Linear Regression
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
