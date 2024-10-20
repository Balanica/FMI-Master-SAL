from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, contamination=0.1)
normal = X_train[y_train == 0]
outliers = X_train[y_train == 1]

plt.scatter(normal[:, 0], normal[:, 1], color='blue')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red')
plt.show()