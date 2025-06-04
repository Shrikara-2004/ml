import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Load Titanic dataset

df1 = sns.load_dataset('titanic')


columns_remove = ['deck', 'embarked', 'alive', 'pclass']
df = df1.drop(columns=columns_remove)


df['age'].fillna(round(df['age'].mean()), inplace=True)  # Fill missing age with mean
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)  # Fill missing embark_town with mode


plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='age', y='fare')
plt.title('Scatter Plot: Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()


correlation = df['age'].corr(df['fare'])
print(f"\nPearson Correlation (age vs fare): {correlation:.2f}")


print("\nCovariance Matrix:")
print(df[['age', 'fare']].cov())


print("\nCorrelation Matrix:")
print(df[['age', 'fare', 'sibsp', 'parch']].corr())


plt.figure(figsize=(6, 5))
sns.heatmap(df[['age', 'fare', 'sibsp', 'parch']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap: Correlation Matrix')
plt.show()
