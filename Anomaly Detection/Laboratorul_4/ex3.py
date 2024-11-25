from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ocsvm import OCSVM
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1
data = loadmat("C:/Users/Andrei/Downloads/shuttle.mat")
y = data['y'].flatten()
X_train, X_test, y_train, y_test = train_test_split(data["X"], y, test_size=0.5)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#2
ocsvm = OCSVM()
ocsvm.fit(X_train)

pred_test_ocsvm = ocsvm.predict(X_test)
ba_ocsvm = balanced_accuracy_score(y_test, pred_test_ocsvm)

score_ocsvm = ocsvm.decision_function(X_test)
roc_ocsvm = roc_auc_score(y_test, score_ocsvm)
print("Balanced Accuracy score for OCSVM: " + str(ba_ocsvm))
print("ROC AUC score for OCSVM: " + str(roc_ocsvm))

n_features = X_train.shape[1]
print(n_features)
dSVDD = DeepSVDD(n_features=n_features)
dSVDD.fit(X_train)

pred_test = dSVDD.predict(X_test)
ba_ocsvm = balanced_accuracy_score(y_test, pred_test)

score_ocsvm = dSVDD.decision_function(X_test)
roc_ocsvm = roc_auc_score(y_test, score_ocsvm)
print("Balanced Accuracy score DeepSVDD: " + str(ba_ocsvm))
print("ROC AUC score for DeepSVDD: " + str(roc_ocsvm))

#3
arhitectures = [[128, 64], [128, 64, 32], [64, 32], [256, 128, 64]]

for arhitecture in arhitectures:
    dSVDD = DeepSVDD(n_features=n_features, hidden_neurons=arhitecture)
    dSVDD.fit(X_train)

    pred_test = dSVDD.predict(X_test)
    ba_ocsvm = balanced_accuracy_score(y_test, pred_test)

    score_ocsvm = dSVDD.decision_function(X_test)
    roc_ocsvm = roc_auc_score(y_test, score_ocsvm)
    print(f"Balanced Accuracy score for DeepSVDD arhitecture {arhitecture}:" + str(ba_ocsvm))
    print(f"ROC AUC score for DeepSVDD arhitecture {arhitecture}: " + str(roc_ocsvm))

