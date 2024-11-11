from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from  sklearn.metrics import roc_auc_score
import numpy as np


data = loadmat("C:/Users/Andrei/Downloads/shuttle.mat")
#print(data)

y = data['y'].flatten()
ba_iforest = []
roc_iforest = []
ba_loda = []
roc_loda = []
ba_dif = []
roc_dif = []

for i in range(2):
    X_train, X_test, y_train, y_test = train_test_split(data["X"], y, test_size=0.4)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    iforest = IForest()
    loda = LODA()
    dif = DIF()
    print(1)
    iforest.fit(X_train)
    loda.fit(X_train)
    dif.fit(X_train)

    predict_iforest = iforest.predict(X_test)
    predict_loda = loda.predict(X_test)
    predict_dif = dif.predict(X_test)
    print(1)
    ba_iforest.append(balanced_accuracy_score(y_test, predict_iforest))
    ba_loda.append(balanced_accuracy_score(y_test, predict_loda))
    ba_dif.append(balanced_accuracy_score(y_test, predict_dif))
    print(1)
    score_iforest = iforest.decision_function(X_test)
    score_loda = loda.decision_function(X_test)
    score_dif = dif.decision_function(X_test)
    print(1)
    roc_iforest.append(roc_auc_score(y_test, score_iforest))
    roc_loda.append(roc_auc_score(y_test, score_loda))
    roc_dif.append(roc_auc_score(y_test, score_dif))

print("Balanced Accuracy mean for IForest: " + str(np.mean(ba_iforest)))
print("Balanced Accuracy mean for DIF: " + str(np.mean(ba_dif)))
print("Balanced Accuracy mean for LODA: " + str(np.mean(ba_loda)))

print("ROC AUC mean for IForest: " + str(np.mean(roc_iforest)))
print("ROC AUC mean for DIF: " + str(np.mean(roc_dif)))
print("ROC AUC mean for LODA: " + str(np.mean(roc_loda)))


