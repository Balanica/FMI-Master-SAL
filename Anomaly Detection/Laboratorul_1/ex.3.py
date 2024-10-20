from pyod.utils.data import generate_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score

X_train, X_test, y_train, y_test = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)

mean = np.mean(X_train)
std = np.std(X_train)

z_scores = (X_train - mean) / std

threshold = np.quantile(z_scores, 0.9)
# print(threshold)
anomali = []
normale = []
prediction = []
for count, x in enumerate(z_scores):
    if abs(x) > threshold:
        anomali.append(X_train[count])
        prediction.append(1)
    else:
        normale.append(X_train[count])
        prediction.append(0)



tn, tp, fn, fp = confusion_matrix(y_train, prediction).ravel()

tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
balanced_accuracy = (tpr + tnr) / 2
# acc = balanced_accuracy_score(y_train, prediction)
# acc2 = accuracy_score(y_train, prediction)
print(balanced_accuracy)

# plt.plot([mean, mean], [5, -5], color="green")
# # plt.plot([mean+threshold, mean+threshold], [5, -5], color="purple")
# # plt.plot([-threshold+mean, -threshold+mean], [5, -5], color="purple")
# plt.scatter(normale,len(normale) * [1], color="blue")
# plt.scatter(anomali,len(anomali) * [1], color="red")
# plt.show()