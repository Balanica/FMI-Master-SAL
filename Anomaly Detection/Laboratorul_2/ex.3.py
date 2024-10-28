import numpy as np
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import matplotlib.pyplot as plt
import pandas as pd
x_1, y_1 = make_blobs(n_samples=200, n_features=2, center_box=(-10, -10), cluster_std=2, centers=1)
x_2, y_2 = make_blobs(n_samples=100, n_features=2, center_box=(10, 10), cluster_std=6, centers=1)

# print(y_1)
# plt.scatter(x_1[:, 0], x_1[:, 1])
# plt.scatter(x_2[:, 0], x_2[:, 1])
# plt.show()

knn = KNN(contamination=0.07, n_neighbors=20)
lof = LOF(contamination=0.07)

combined_data = pd.DataFrame(x_1)
x_2_pd = pd.DataFrame(x_2)
combined_data = combined_data._append(x_2_pd)

knn.fit(combined_data)
lof.fit(combined_data)
pred_knn = knn.predict(combined_data)
pred_lof = lof.predict(combined_data)

combined_data = np.array(combined_data)

inliers_pred_knn = combined_data[pred_knn == 0]
outliers_pred_knn = combined_data[pred_knn == 1]
inliers_pred_lof = combined_data[pred_lof == 0]
outliers_pred_lof = combined_data[pred_lof == 1]

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].scatter(inliers_pred_knn[:, 0], inliers_pred_knn[:, 1], color='blue')
ax[0].scatter(outliers_pred_knn[:, 0], outliers_pred_knn[:, 1], color='red')
ax[0].title.set_text('Prediction KNN')

ax[1].scatter(inliers_pred_lof[:, 0], inliers_pred_lof[:, 1], color='blue')
ax[1].scatter(outliers_pred_lof[:, 0], outliers_pred_lof[:, 1], color='red')
ax[1].title.set_text('Prediction LOF')

plt.show()