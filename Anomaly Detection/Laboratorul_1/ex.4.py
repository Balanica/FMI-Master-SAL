from pyod.utils.data import generate_data
import numpy as np
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = generate_data(n_train=1000, n_test=0, n_features=4, contamination=0.1)

mean = np.mean(X_train, 0)
std = np.std(X_train, 0)
z_scores = (X_train - mean) / std
total_z_scores = np.linalg.norm(z_scores, axis=1)
threshold = np.quantile(z_scores, 0.9)

anomali = []
normale = []
prediction = []
for count, x in enumerate(total_z_scores):
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
print(balanced_accuracy)
