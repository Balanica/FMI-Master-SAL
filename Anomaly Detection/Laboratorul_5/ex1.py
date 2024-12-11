from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np

points = multivariate_normal(mean=[5, 10, 2], cov=[[3, 2, 2], [2, 10, 1], [2, 1, 2]], size=500)

# 1
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2])
plt.show()

centered_points = points - np.mean(points, axis=0)
cov_matrix = np.cov(centered_points.T)

eigvals, eigvecs = np.linalg.eigh(cov_matrix)

# 2
desc_sorted_indices = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[desc_sorted_indices]
eigvecs_sorted = eigvecs[:, desc_sorted_indices]

individual_variance = eigvals_sorted / np.sum(eigvals_sorted)
cumulative_variance = np.cumsum(individual_variance)

fig, ax = plt.subplots()
ax.step(range(1, len(eigvals_sorted) + 1), cumulative_variance)
ax.bar(range(1, len(eigvals_sorted) + 1), individual_variance)
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.title('Explained Variance Individual/Cumulative')
plt.show()

# 3
projected_points = np.dot(centered_points, eigvecs_sorted)
threshold_2 = np.quantile(projected_points[:, 1], 0.9)
threshold_3 = np.quantile(projected_points[:, 2], 0.9)

inliers = points[projected_points[:, 2] > threshold_3]
outliers = points[projected_points[:, 2] <= threshold_3]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color="blue")
ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color="red")
ax.set_title("3rd Principal Component")
plt.show()

inliers = points[projected_points[:, 1] > threshold_2]
outliers = points[projected_points[:, 1] <= threshold_2]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color="blue")
ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color="red")
ax.set_title("2rd Principal Component")
plt.show()

# 4
centroid = np.mean(projected_points, axis=0)

dist = np.linalg.norm(projected_points - centroid, axis=1)
norm_dist = dist / np.std(dist)

threshold = np.quantile(norm_dist, 0.9)

inliers = points[norm_dist > threshold]
outliers = points[norm_dist <= threshold]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], color="blue")
ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], color="red")
ax.set_title("Anomaly detection based on Normalized Distance")
plt.show()
