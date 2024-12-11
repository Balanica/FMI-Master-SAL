import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
import keras
import matplotlib.pyplot as plt

# 1
data = loadmat("C:/Users/Andrei/Downloads/shuttle.mat")
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2
class Autoencoder(keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(3, activation='relu')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(9, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')
model = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test, X_test),
    verbose=1
)
# 3
plt.plot(model.history['loss'], label='Training Loss')
plt.plot(model.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 4
train_model = autoencoder(X_train)
test_model = autoencoder(X_test)

train_errors = np.mean(np.square(X_train-train_model), axis=1)
test_errors = np.mean(np.square(X_test-test_model), axis=1)

contamination_rate = np.mean(y_train)
threshold = np.quantile(train_errors, 1-contamination_rate)

train_pred = (train_errors > threshold).astype(int)
test_pred = (test_errors > threshold).astype(int)
train_bal_acc = balanced_accuracy_score(y_train, train_pred)
test_bal_acc = balanced_accuracy_score(y_test, test_pred)

print("Balanced Accuracy Train: " + str(train_bal_acc))
print("Balanced Accuracy Test: " + str(test_bal_acc))