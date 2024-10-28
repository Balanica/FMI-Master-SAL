from pyod.models.knn import KNN
from pyod.utils.data import generate_data_clusters
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1)

inliers_train = X_train[y_train == 0]
outliers_train = X_train[y_train == 1]
inliers_test = X_test[y_test == 0]
outliers_test = X_test[y_test == 1]

classfier = KNN(n_neighbors=20)

classfier.fit(X_train, y_train)
pred_train = classfier.predict(X_train)
pred_test = classfier.predict(X_test)

inliers_pred_train = X_train[pred_train == 0]
outliers_pred_train = X_train[pred_train == 1]
inliers_pred_test = X_test[pred_test == 0]
outliers_pred_test = X_test[pred_test == 1]

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0, 0].scatter(inliers_train[:, 0], inliers_train[:, 1], color='blue')
ax[0, 0].scatter(outliers_train[:, 0], outliers_train[:, 1], color='red')
ax[0, 0].title.set_text('Ground truth labels for training data')

ax[1, 0].scatter(inliers_test[:, 0], inliers_test[:, 1], color='blue')
ax[1, 0].scatter(outliers_test[:, 0], outliers_test[:, 1], color='red')
ax[1, 0].title.set_text('Ground truth labels for test data')

ax[0, 1].scatter(inliers_pred_train[:, 0], inliers_pred_train[:, 1], color='blue')
ax[0, 1].scatter(outliers_pred_train[:, 0], outliers_pred_train[:, 1], color='red')
ax[0, 1].title.set_text('Predicted labels for training data')

ax[1, 1].scatter(inliers_pred_test[:, 0], inliers_pred_test[:, 1], color='blue')
ax[1, 1].scatter(outliers_pred_test[:, 0], outliers_pred_test[:, 1], color='red')
ax[1, 1].title.set_text('Predicted labels for test data')
plt.show()