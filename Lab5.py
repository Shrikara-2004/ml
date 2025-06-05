import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load Iris dataset
iris = sns.load_dataset("iris")


iris.hist(figsize=(8, 5), color='skyblue')
plt.suptitle("Histogram of Iris Features")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=iris)
plt.title("Boxplot of Iris Features")
plt.xticks(rotation=45)
plt.show()


X = iris.drop(columns=["species"])   # Features
y = iris["species"]                  # Target labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Regular k-NN Results:")
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"\nk = {k}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
    print("F1-score:", round(f1_score(y_test, y_pred, average='macro'), 2))


print("\nWeighted k-NN Results (weights = 1/d²):")
for k in [1, 3, 5]:
    weighted_knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    weighted_knn.fit(X_train, y_train)
    y_pred_w = weighted_knn.predict(X_test)
    print(f"\nk = {k}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred_w), 2))
    print("F1-score:", round(f1_score(y_test, y_pred_w, average='macro'), 2))
