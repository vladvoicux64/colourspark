import keras
import matplotlib.pyplot as plt
from keras.src.layers import LeakyReLU
from keras.src.optimizers import Adam

import data_utils as du
from keras.src.callbacks import TensorBoard
from stl10_input import read_all_images
import os

TRAIN_DATA_PATH = './data/stl10_binary/unlabeled_X.bin'
VALIDATION_DATA_PATH = './data/stl10_binary/train_X.bin'
TEST_DATA_PATH = './data/stl10_binary/test_X.bin'
IMG_SIZE = (96, 96)
BATCH_SIZE = 8

if __name__ == "__main__":
    # read images from disk
    train_images = read_all_images(TRAIN_DATA_PATH)

    validation_images = read_all_images(VALIDATION_DATA_PATH)

    test_images = read_all_images(TEST_DATA_PATH)

    # create datasets
    train_ds_x = du.convert_to_grayscale(train_images).astype('float32') / 255

    train_ds_y = train_images.astype('float32') / 255

    validation_ds_x = du.convert_to_grayscale(validation_images).astype('float32') / 255

    validation_ds_y = validation_images.astype('float32') / 255

    test_ds_x = du.convert_to_grayscale(test_images).astype('float32') / 255

    test_ds_y = test_images.astype('float32') / 255

    # model
    input_img = keras.Input(IMG_SIZE + (1,))

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

    autoencoder.compile(optimizer=Adam(0.0005), loss='mse')

    autoencoder.fit(train_ds_x, train_ds_y,
                    epochs=1,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(validation_ds_x, validation_ds_y),
                    callbacks=[TensorBoard(log_dir='/tmp/tb')])

    autoencoder.save('autoencoder.keras')

    decoded_imgs = autoencoder.predict(test_ds_x[:10])

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(test_ds_x[i - 1].reshape(96, 96))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i - 1].reshape(96, 96, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
