import numpy as np
import matplotlib.pyplot as plt

class KMeansScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def initialize_centroids(self, X):
        # Randomly choose k points from dataset
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]

    def compute_distance(self, X, centroids):
        # Euclidean distance
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return distances

    def assign_clusters(self, distances):
        # Choose closest centroid
        return np.argmin(distances, axis=0)

    def update_centroids(self, X, labels):
        new_centroids = []
        for i in range(self.k):
            points = X[labels == i]
            if len(points) == 0:
                new_centroids.append(self.centroids[i])
            else:
                new_centroids.append(points.mean(axis=0))
        return np.array(new_centroids)

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iters):
            distances = self.compute_distance(X, self.centroids)
            labels = self.assign_clusters(distances)
            new_centroids = self.update_centroids(X, labels)

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return labels

    def predict(self, X):
        distances = self.compute_distance(X, self.centroids)
        return self.assign_clusters(distances)
# test the implementation
# DATA
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6,
               7, 8, 9, 8, 9, 9, 8, 4, 4, 5, 4])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7,
               1, 2, 1, 2, 3, 2, 3, 9, 10, 9, 10])

X = np.array(list(zip(x1, x2)))

# TRAIN
kmeans = KMeansScratch(k=3)
labels = kmeans.fit(X)

# PLOT
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
            c='red', s=300, label='Centroids')

plt.title("KMeans from Scratch")
plt.legend()
plt.grid()
plt.show()