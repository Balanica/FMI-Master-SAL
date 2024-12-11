from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
import matplotlib.pyplot as plt
import numpy as np
# 1
data = loadmat("C:/Users/Andrei/Downloads/shuttle.mat")
y = data['y'].flatten()
X_train, X_test, y_train, y_test = train_test_split(data["X"], y, test_size=0.4)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

contamination_rate = y_train.mean()

pca = PCA(contamination=contamination_rate)
pca.fit(X_train)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure()
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label="Cumulative Variance")
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.title('Explained Variance Individual/Cumulative')
plt.show()


# 2

y_train_pred = pca.predict(X_train)
y_test_pred = pca.predict(X_test)
train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)

print("PCA Balanced Accuracy Train: " + str(train_bal_acc))
print("PCA Balanced Accuracy Test: " + str(test_bal_acc))

data = loadmat("C:/Users/Andrei/Downloads/shuttle.mat")
y = data['y'].flatten()
X_train, X_test, y_train, y_test = train_test_split(data["X"], y, test_size=0.4)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

contamination_rate = y_train.mean()


kpca = KPCA(contamination=contamination_rate)
kpca.fit(X_train)

y_train_pred_kpca = kpca.predict(X_train)
y_test_pred_kpca = kpca.predict(X_test)
train_bal_acc_kpca = balanced_accuracy_score(y_train, y_train_pred_kpca)
test_bal_acc_kpca = balanced_accuracy_score(y_test, y_test_pred_kpca)

print("KPCA Balanced Accuracy Train: " + str(train_bal_acc_kpca))
print("KPCA Balanced Accuracy Test: " + str(test_bal_acc_kpca))

