from scipy.io import loadmat
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer

# 1
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

data = loadmat("C:/Users/Andrei/Downloads/cardio.mat")
y = data['y'].flatten()
y = -(2 * y - 1)
X_train, X_test, y_train, y_test = train_test_split(data["X"], y, test_size=0.6)

# 2
grid = {
    "classifier__kernel": ["rbf", "linear", "poly", "sigmoid"],
    "classifier__gamma": ["scale", "auto", 0.1, 0.01, 0.001],
    "classifier__nu": [0.1, 0.3, 0.5, 0.7]
}

# 4
pipeline = Pipeline([("scalar", StandardScaler()), ("classifier", OneClassSVM())])

# 3
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=grid,
    scoring=make_scorer(balanced_accuracy_score),
    cv=5
)

# 6
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
results = grid_search.cv_results_
print(f"Best parameters from the grid: {grid_search.best_params_}")
print(f"The best score from the grid: {grid_search.best_score_} ")

y_pred = best_model.predict(X_test)
ba_score = balanced_accuracy_score(y_test, y_pred)

print("Balanced accuracy score for the best model: " + str(ba_score))
