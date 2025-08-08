#(10).Write a program to implement clustering using the k-Means algorithm using an appropriate dataset. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Load dataset and apply K-Means
X, y = load_iris(return_X_y=True)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y, y_kmeans))

# PCA for 2D visualization
X_pca = PCA(n_components=2).fit_transform(X)
colors = ['red', 'green', 'blue']

# Plot clusters
for i, color in enumerate(colors):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], 
                 label=f"Cluster {i}", c=color, edgecolor='k')

plt.title('K-Means Clustering (Iris Dataset)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.show()