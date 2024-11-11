from sklearn.datasets import make_blobs
from numpy.random import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=500, n_features=2, cluster_std=1.0, centers=1)

projections_vectors = multivariate_normal(mean=(0, 0), cov=[[1, 0], [0, 1]], size=5)
projections = []
histograms = []
range_bin = (-20, 20)
nr_vec = 5
nr_bins = 100 # 10 , 30 , 60 , 80
for i in range(nr_vec):
    projections_vectors[i] = projections_vectors[i] / np.linalg.norm(projections_vectors[i])
    projections.append(X @ projections_vectors[i])
    histogram, bins = np.histogram(projections[i], range=range_bin, bins = nr_bins, density=True)
    bin_prob = histogram / np.sum(histogram)
    print(bin_prob)
    histograms.append((bin_prob, bins))

probabilities = []
for i in range(nr_vec):
    proj = X @ projections_vectors[i]
    bins = np.digitize(proj, histograms[i][1])
    # print(bins)
    prob = []
    for j in range(len(X)):
        prob.append(histograms[i][0][bins[j]])
    # print(prob)
    probabilities.append(prob)

mean_train = np.mean(probabilities, axis=0)
print(mean_train)
plt.scatter(X[:, 0], X[:, 1], c=mean_train, cmap='viridis')
plt.colorbar()
plt.show()

X_test = np.random.uniform(-3, 3, (500, 2))

probabilities_test = []
for i in range(nr_vec):
    proj = X_test @ projections_vectors[i]
    bins = np.digitize(proj, histograms[i][1])
    # print(bins)
    prob = []
    for j in range(len(X)):
        prob.append(histograms[i][0][bins[j]])
    # print(prob)
    probabilities_test.append(prob)

mean_test = np.mean(probabilities_test, axis=0)
print(mean_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c=mean_test, cmap='viridis')
plt.colorbar()
plt.show()