import numpy as np
from scipy.io import loadmat
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

data = loadmat("C:/Users/Andrei/Downloads/cardio.mat")
print(data["y"])
y = data['y'].flatten()
X_train, X_test, y_train, y_test = train_test_split(data["X"], y, test_size=0.3)
X_train = standardizer(X_train)
X_test = standardizer(X_test)
# print(X_train)
contamination = len(data["y"][data["y"] == 1])/len(data["y"])
print(contamination)
knn_train_scores = []
lof_train_scores = []
knn_test_scores = []
lof_test_scores = []

for i in range(10):
    knn = KNN(n_neighbors=30+9*i, contamination=contamination)
    lof = LOF(n_neighbors=30+9*i, contamination=contamination)
    print("For " + str(30+9*i) + " neighbors:")
    knn.fit(X_train)
    lof.fit(X_train)

    knn_pred_train = knn.predict(X_train)
    knn_pred_test = knn.predict(X_test)
    lof_pred_train = lof.predict(X_train)
    lof_pred_test = lof.predict(X_test)

    knn_ba_train = balanced_accuracy_score(y_train, knn_pred_train)
    knn_ba_test = balanced_accuracy_score(y_test, knn_pred_test)

    lof_ba_train = balanced_accuracy_score(y_train, lof_pred_train)
    lof_ba_test = balanced_accuracy_score(y_test, lof_pred_test)

    print("Balanced accuracy train KNN:" + str(knn_ba_train))
    print("Balanced accuracy test KNN:" + str(knn_ba_test) + "\n")

    print("Balanced accuracy train LOF:" + str(lof_ba_train))
    print("Balanced accuracy test LOF:" + str(lof_ba_test) + "\n")

    knn_train_scores.append(knn.decision_scores_)
    lof_train_scores.append(lof.decision_scores_)
    knn_test_scores.append(knn.decision_function(X_test))
    lof_test_scores.append(lof.decision_function(X_test))

# print(knn_train_scores)
# print(knn_test_scores)
knn_train_scores = np.column_stack(knn_train_scores)
lof_train_scores = np.column_stack(lof_train_scores)

# print(np.shape(knn_train_scores))
# print(np.shape(knn_test_scores))

knn_st_score = standardizer(knn_train_scores)
lof_st_score = standardizer(lof_train_scores)

# print(knn_st_score)
# print(lof_st_score)

knn_avg_score = average(knn_st_score)
lof_avg_score = average(lof_st_score)
knn_max_score = maximization(knn_st_score)
lof_max_score = maximization(lof_st_score)

knn_test_scores = np.column_stack(knn_test_scores)
lof_test_scores = np.column_stack(lof_test_scores)

knn_st_test_score = standardizer(knn_test_scores)
lof_st_test_score = standardizer(lof_test_scores)

knn_avg_test_score = average(knn_st_test_score)
lof_avg_test_score = average(lof_st_test_score)
knn_max_test_score = maximization(knn_st_test_score)
lof_max_test_score = maximization(lof_st_test_score)

print(knn_avg_score)
print(lof_avg_score)
# print(knn_max_score)
# print(lof_max_score)

knn_avg_threshold = np.quantile(knn_avg_score, 1-contamination)
lof_avg_threshold = np.quantile(knn_avg_score, 1-contamination)
knn_max_threshold = np.quantile(knn_max_score, 1-contamination)
lof_max_threshold = np.quantile(knn_max_score, 1-contamination)

knn_avg_test_threshold = np.quantile(knn_avg_test_score, 1-contamination)
lof_avg_test_threshold = np.quantile(knn_avg_test_score, 1-contamination)
knn_max_test_threshold = np.quantile(knn_max_test_score, 1-contamination)
lof_max_test_threshold = np.quantile(knn_max_test_score, 1-contamination)

# print(knn_avg_threshold)
# print(lof_avg_threshold)
# print(knn_max_threshold)
# print(lof_max_threshold)
#
# print(knn_avg_test_threshold)
# print(lof_avg_test_threshold)
# print(knn_max_test_threshold)
# print(lof_max_test_threshold)

knn_predictii = (knn_avg_score >= knn_avg_threshold).astype(int)
lof_predictii = (lof_avg_score >= lof_avg_threshold).astype(int)
knn_predictii_test = (knn_avg_test_score >= knn_avg_test_threshold).astype(int)
lof_predictii_test = (lof_avg_test_score >= lof_avg_test_threshold).astype(int)


knn_ba_avg = balanced_accuracy_score(y_train, knn_predictii)
lof_ba_avg = balanced_accuracy_score(y_train, lof_predictii)
knn_ba_avg_test = balanced_accuracy_score(y_test, knn_predictii_test)
lof_ba_avg_test = balanced_accuracy_score(y_test, lof_predictii_test)


print("Balanced accuracy for KNN combination train: " + str(knn_ba_avg))
print("Balanced accuracy for LOF combination train: " + str(lof_ba_avg))
print("Balanced accuracy for KNN combination test: " + str(knn_ba_avg_test))
print("Balanced accuracy for LOF combination test: " + str(lof_ba_avg_test))