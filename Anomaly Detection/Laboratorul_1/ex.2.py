from pyod.models.knn import KNN
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from  sklearn.metrics import roc_curve

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, contamination=0.1)
normal = X_train[y_train == 0]
outliers = X_train[y_train == 1]

knn = KNN(contamination=0.2)
knn.fit(X_train)

train_pred = knn.predict(X_train)
test_pred = knn.predict(X_test)


tn, tp, fn, fp = confusion_matrix(y_train, train_pred).ravel()

print(tn, tp, fn, tp)

recall = tp / (tp+fn)
specificity = tn / (tn + fp)
balanced_accuracy = (recall + specificity) / 2
print(balanced_accuracy)

fpr, tpr, thresholds = roc_curve(y_test, knn.decision_function(X_test))

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.show()