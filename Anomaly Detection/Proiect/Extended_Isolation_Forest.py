import numpy as np
from scipy.io import loadmat
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from  sklearn.metrics import roc_auc_score

e = 0.5772156649


class Extended_Isolation_Tree:
    def __init__(self, X, max_height, height=0):
        self.X = X
        self.max_height = max_height
        self.height = height
        self.left_tree = None
        self.right_tree = None
        self.normal = None
        self.intercept = None
        self.end = 0
        # print(X.shape[0])
        if height >= max_height or X.shape[0] <= 1:
            self.end = 1
            return
        else:
            self.normal = np.random.normal(0, 1, X.shape[1])
            min_feature = np.min(X, axis=0)
            max_feature = np.max(X, axis=0)
            self.intercept = np.random.uniform(min_feature, max_feature)
            self.left_tree = Extended_Isolation_Tree(X[np.dot(X - self.intercept, self.normal) < 0], self.max_height, self.height + 1)
            self.right_tree = Extended_Isolation_Tree(X[np.dot(X - self.intercept, self.normal) >= 0], self.max_height,
                                             self.height + 1)

    def path_length(self, data, length=0):
        if self.end:
            if self.X.shape[0]<=1:
                return length
            else:
                return length + 2 * (np.log(self.X.shape[0]) + e) - (2 * (self.X.shape[0] - 1) / self.X.shape[0])
        # print(np.dot(data - self.intercept, self.normal), self.intercept, self.normal)
        if np.dot(data - self.intercept, self.normal) < 0:
            return self.left_tree.path_length(data, length + 1)
        else:
            return self.right_tree.path_length(data, length + 1)


class Extended_Isolation_Forest:
    def __init__(self, max_height=None, sub_sample_size=256, tree_number=100, contamination=0):
        self.max_height = max_height
        self.sub_sample_size = sub_sample_size
        self.tree_number = tree_number
        self.contamination = contamination
        self.isolation_forest = []

    def fit(self, X):
        self.max_height = np.log(X.shape[0])
        for i in range(self.tree_number):
            sub_sample = X[np.random.choice(X.shape[0], self.sub_sample_size, replace=False)]
            isolation_tree = Extended_Isolation_Tree(sub_sample, self.max_height)
            self.isolation_forest.append(isolation_tree)

    def anomaly_score(self, X):
        lengths_X = []
        for tree in self.isolation_forest:
            lengths_X.append([tree.path_length(x) for x in X])
        lengths_X = np.array(lengths_X)
        average_lengths = np.mean(lengths_X, axis=0)
        score = 2 ** (-average_lengths / (2 * (np.log(X.shape[0]) + e) - (2 * (X.shape[0] - 1) / X.shape[0])))
        # print(average_lengths, (2 * (np.log(X.shape[0]) + e) - (2 * (X.shape[0] - 1) / X.shape[0])))
        return score

    def predict(self, X):
        score = self.anomaly_score(X)
        threshold = 0
        if self.contamination:
            threshold = np.quantile(score, 1-self.contamination)
        else:
            threshold = 0.7
        #print(threshold)
        #print(np.mean(score))
        pred = [int(score[i] > threshold) for i in range(X.shape[0])]
        return pred


# X1, y1 = make_blobs(n_samples=1000, n_features=2, cluster_std=1.0, centers=2, center_box=((0, 10), (10, 0)))
#
# print(y1)
# cluster_1 = np.random.randn(5000, 2) * 1.0 + [0, 10]  # Centered at (0,10)
# cluster_2 = np.random.randn(5000, 2) * 1.0 + [10, 0]  # Centered at (10,0)
# X = np.vstack((cluster_1, cluster_2))
# ceva = Extended_Isolation_Forest(1000, contamination=0.01)
# ceva.fit(X)
# ud = np.random.uniform(low=-10, high=20, size=(2000, 2))
# scores2 = ceva.predict(X1)
#
#
# fig, ax = plt.subplots()
# scatter = ax.scatter(X1[:, 0],X1[:, 1], c=scores2, cmap=plt.cm.RdYlBu)
# cbar = plt.colorbar(scatter, ax=ax)
# cbar.set_label('Anomaly Score')
# plt.show()
#
# cluster_1 = np.random.randn(5000, 2) * 1.0 + [5, 5]  # Centered at (0,10)
# X = cluster_1
# ceva = Extended_Isolation_Forest(1000, contamination=0.01)
# ceva.fit(X)
# ud = np.random.uniform(low=-10, high=20, size=(2000, 2))
# scores2 = ceva.anomaly_score(ud)
#
# fig, ax = plt.subplots()
# scatter = ax.scatter(ud[:, 0], ud[:, 1], c=scores2, cmap=plt.cm.RdYlBu)
# cbar = plt.colorbar(scatter, ax=ax)
# cbar.set_label('Anomaly Score')
# plt.show()


# data = loadmat("C:/Users/Andrei/Downloads/shuttle.mat")
# data = loadmat("C:/Users/Andrei/Downloads/cardio.mat")
# data = loadmat("C:/Users/Andrei/Downloads/arrhythmia.mat")
# y = data['y'].flatten()
# X_train, X_test, y_train, y_test = train_test_split(data["X"], y, test_size=0.4)
# data = np.loadtxt("C:/Users/Andrei/Downloads/satellite.txt", delimiter=",")
data = np.loadtxt("C:/Users/Andrei/Downloads/mammography.txt", delimiter=",")
X = data[:, :-1]  # All rows, all columns except last
y = data[:, -1]   # All rows, only last column

# X_train, X_test, y_train, y_test = train_test_split(data["X"], y, test_size=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
cont = np.mean(y)
print(cont)
ceva = Extended_Isolation_Forest(1000, contamination=cont, sub_sample_size=128)
ceva.fit(X_train)

pred = ceva.predict(X_train)
pred_test = ceva.predict(X_test)
print(balanced_accuracy_score(y_train, pred))
print(balanced_accuracy_score(y_test, pred_test))

score = ceva.anomaly_score(X_train)
score1 = ceva.anomaly_score(X_test)
print(roc_auc_score(y_train, score))
print(roc_auc_score(y_test, score1))