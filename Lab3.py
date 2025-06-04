from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Labels (0, 1, 2)

# Apply PCA to reduce 4D to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Print Eigenvalues (explained variance)
print("Eigenvalues (Explained Variance):")
print(pca.explained_variance_)

# Print Eigenvectors (Principal Components)
print("\nEigenvectors (Principal Components):")
print(pca.components_)

# Plot the PCA result
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
