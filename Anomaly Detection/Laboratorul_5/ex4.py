import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
# 1
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


X_train_noise = tf.clip_by_value(X_train + tf.random.normal(shape=X_train.shape) * 0.35, 0.0, 1.0)
X_test_noise = tf.clip_by_value(X_test + tf.random.normal(shape=X_test.shape) * 0.35, 0.0, 1.0)

# 2
class ConvolutionalAutoencoder(keras.Model):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(8, (3, 3), activation='relu', strides=2, padding='same'),
            keras.layers.Conv2D(4, (3, 3), activation='relu', strides=2, padding='same')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(4, (3, 3), activation='relu', strides=2, padding='same'),
            keras.layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same'),
            keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# 3
autoencoder = ConvolutionalAutoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

model = autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, X_test),
    verbose=2
)

train_model = autoencoder(X_train)
test_model = autoencoder(X_test)
train_noise_model = autoencoder(X_train_noise)
test_noise_model = autoencoder(X_test_noise)

train_errors = np.mean(np.square(X_train - train_model), axis=(1, 2, 3))
test_errors = np.mean(np.square(X_test - test_model), axis=(1, 2, 3))
train_noise_errors = np.mean(np.square(X_train_noise - train_noise_model), axis=(1, 2, 3))
test_noise_errors = np.mean(np.square(X_test_noise - test_noise_model), axis=(1, 2, 3))

threshold = np.mean(train_errors) + np.std(train_errors)

test_pred = (test_errors > threshold).astype(int)
test_noise_pred = (test_noise_errors > threshold).astype(int)

test_acc = 1 - test_pred.mean()
test_noise_acc = 1 - test_noise_pred.mean()

print("Balanced Accuracy Clean Test: " + str(test_acc))
print("Balanced Accuracy Noise Test: " + str(test_noise_acc))

# 4
plt.figure()

for i in range(5):
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 5, i + 6)
    plt.imshow(X_test_noise[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 5, i + 11)
    plt.imshow(test_model[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 5, i + 16)
    plt.imshow(test_noise_model[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')

plt.show()

# 5
model = autoencoder.fit(
    X_train_noise, X_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_noise, X_test),
    verbose=2
)

denoise_test_model = autoencoder(X_test)
denoise_test_noise_model = autoencoder(X_test_noise)

for i in range(5):
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 5, i + 6)
    plt.imshow(X_test_noise[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 5, i + 11)
    plt.imshow(denoise_test_model[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(4, 5, i + 16)
    plt.imshow(denoise_test_noise_model[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')

plt.show()