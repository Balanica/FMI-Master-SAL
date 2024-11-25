from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 1
X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=300, n_features=3, contamination=0.15)


def OCSVM_training(kernel='linear'):
    # 2
    ocsvm = OCSVM(contamination=0.15, kernel=kernel)
    ocsvm.fit(X_train)
    pred_test_ocsvm = ocsvm.predict(X_test)
    pred_train_ocsvm = ocsvm.predict(X_train)
    ba_ocsvm = balanced_accuracy_score(y_test, pred_test_ocsvm)

    score_ocsvm = ocsvm.decision_function(X_test)
    roc_ocsvm = roc_auc_score(y_test, score_ocsvm)
    print("Balanced Accuracy score: " + str(ba_ocsvm))
    print("ROC AUC score: " + str(roc_ocsvm))

    # 3
    inliers_train = X_train[y_train == 0]
    outliers_train = X_train[y_train == 1]
    fig, ax = plt.subplots(nrows=2, ncols=2, subplot_kw={"projection": "3d"})
    fig.suptitle("OCSVM with kernel: " + kernel)
    ax[0, 0].scatter(inliers_train[:, 0], inliers_train[:, 1], inliers_train[:, 2], color='blue')
    ax[0, 0].scatter(outliers_train[:, 0], outliers_train[:, 1], outliers_train[:, 2], color='red')
    ax[0, 0].title.set_text('Ground truth train')
    inliers_test = X_test[y_test == 0]
    outliers_test = X_test[y_test == 1]
    ax[0, 1].scatter(inliers_test[:, 0], inliers_test[:, 1], inliers_test[:, 2], color='blue')
    ax[0, 1].scatter(outliers_test[:, 0], outliers_test[:, 1], outliers_test[:, 2], color='red')
    ax[0, 1].title.set_text('Ground truth test')

    inliers_train_pred = X_train[pred_train_ocsvm == 0]
    outliers_train_pred = X_train[pred_train_ocsvm == 1]
    ax[1, 0].scatter(inliers_train_pred[:, 0], inliers_train_pred[:, 1], inliers_train_pred[:, 2], color='blue')
    ax[1, 0].scatter(outliers_train_pred[:, 0], outliers_train_pred[:, 1], outliers_train_pred[:, 2], color='red')
    ax[1, 0].title.set_text('Prediction train')

    inliers_test_pred = X_test[pred_test_ocsvm == 0]
    outliers_test_pred = X_test[pred_test_ocsvm == 1]
    ax[1, 1].scatter(inliers_test_pred[:, 0], inliers_test_pred[:, 1], inliers_test_pred[:, 2], color='blue')
    ax[1, 1].scatter(outliers_test_pred[:, 0], outliers_test_pred[:, 1], outliers_test_pred[:, 2], color='red')
    ax[1, 1].title.set_text('Prediction test')
    plt.show()


# 5
def DeepSVDD_training():
    dSVDD = DeepSVDD(contamination=0.15, n_features=3)
    dSVDD.fit(X_train)
    pred_test = dSVDD.predict(X_test)
    pred_train = dSVDD.predict(X_train)
    ba_ocsvm = balanced_accuracy_score(y_test, pred_test)

    score_ocsvm = dSVDD.decision_function(X_test)
    roc_ocsvm = roc_auc_score(y_test, score_ocsvm)
    print("Balanced Accuracy score: " + str(ba_ocsvm))
    print("ROC AUC score: " + str(roc_ocsvm))

    # 3
    inliers_train = X_train[y_train == 0]
    outliers_train = X_train[y_train == 1]
    print(len(outliers_train))
    fig, ax = plt.subplots(nrows=2, ncols=2, subplot_kw={"projection": "3d"})
    fig.suptitle("DeepSVDD")
    ax[0, 0].scatter(inliers_train[:, 0], inliers_train[:, 1], inliers_train[:, 2], color='blue')
    ax[0, 0].scatter(outliers_train[:, 0], outliers_train[:, 1], outliers_train[:, 2], color='red')
    ax[0, 0].title.set_text('Ground truth train')
    inliers_test = X_test[y_test == 0]
    outliers_test = X_test[y_test == 1]
    ax[0, 1].scatter(inliers_test[:, 0], inliers_test[:, 1], inliers_test[:, 2], color='blue')
    ax[0, 1].scatter(outliers_test[:, 0], outliers_test[:, 1], outliers_test[:, 2], color='red')
    ax[0, 1].title.set_text('Ground truth test')

    inliers_train_pred = X_train[pred_train == 0]
    outliers_train_pred = X_train[pred_train == 1]
    ax[1, 0].scatter(inliers_train_pred[:, 0], inliers_train_pred[:, 1], inliers_train_pred[:, 2], color='blue')
    ax[1, 0].scatter(outliers_train_pred[:, 0], outliers_train_pred[:, 1], outliers_train_pred[:, 2], color='red')
    ax[1, 0].title.set_text('Prediction train')

    inliers_test_pred = X_test[pred_test == 0]
    outliers_test_pred = X_test[pred_test == 1]
    ax[1, 1].scatter(inliers_test_pred[:, 0], inliers_test_pred[:, 1], inliers_test_pred[:, 2], color='blue')
    ax[1, 1].scatter(outliers_test_pred[:, 0], outliers_test_pred[:, 1], outliers_test_pred[:, 2], color='red')
    ax[1, 1].title.set_text('Prediction test')
    plt.show()


OCSVM_training('rbf')
OCSVM_training()
DeepSVDD_training()
