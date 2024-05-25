import keras
import matplotlib.pyplot as plt
import cv2
from keras.src.layers import LeakyReLU
import os

image = cv2.imread('car_at_night.jpg', cv2.IMREAD_GRAYSCALE).astype('float32') / 255
image_as_array = image.reshape((1,) + image.shape + (1,))
input_img = keras.Input(image.shape + (1,))
x = keras.layers.Conv2D(64, (3, 3), activation=LeakyReLU(), padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(128, (3, 3), activation=LeakyReLU(), padding='same')(x)
encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

x = keras.layers.Conv2D(128, (3, 3), activation=LeakyReLU(), padding='same')(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation=LeakyReLU(), padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

if os.path.exists('./autoencoder.keras'):
    autoencoder.load_weights('autoencoder.keras')

decoded_img = autoencoder.predict(image_as_array)

plt.imshow(decoded_img.reshape(decoded_img.shape[1:]))
plt.show()
