import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read dataset
data = pd.read_csv('dataset/digits_data.csv').values
X = data[:, :-1]
y = data[:, -1]

# calculate mean
n_samples = X.shape[0]
X_mean = np.sum(X, axis=0) / n_samples
X_centered = X - X_mean

# calculate covariance matrix
cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

# calculate eigenvalues and eigenvectors of covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# select the first two principal components
principal_components = eigenvectors[:, :2]
print("Projection Matrix:")
print(principal_components)

# project to those principal components
X_pca = X_centered.dot(principal_components)

plt.figure(figsize=(10, 8))
for class_value in np.unique(y):
    plt.scatter(X_pca[y == class_value, 0], X_pca[y == class_value, 1], label=f"Class {int(class_value)}", alpha=0.7)

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA Dimensionality Reduction to 2D")
plt.legend()
plt.grid(True)
plt.show()
