import keras
import matplotlib.pyplot as plt
import data_utils as du
from keras.src.callbacks import TensorBoard
from stl10_input import read_all_images

TRAIN_DATA_PATH = './data/stl10_binary/unlabeled_X.bin'
VALIDATION_DATA_PATH = './data/stl10_binary/train_X.bin'
IMG_SIZE = (96, 96)
BATCH_SIZE = 128
if __name__ == "__main__":
    # read images from disk
    train_images = read_all_images(TRAIN_DATA_PATH)

    validation_images = read_all_images(VALIDATION_DATA_PATH)

    # create datasets
    train_ds_x = du.convert_to_grayscale(train_images).astype('float32') / 255

    train_ds_y = train_images.astype('float32') / 255

    validation_ds_x = du.convert_to_grayscale(validation_images).astype('float32') / 255

    validation_ds_y = validation_images.astype('float32') / 255

    # model
    input_img = keras.Input(IMG_SIZE + (1,))

    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(3, (3, 3), activation='softmax', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(train_ds_x, train_ds_y,
                    epochs=100,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(validation_ds_x, validation_ds_y),
                    callbacks=[TensorBoard(log_dir='/tmp/tb')])

    decoded_imgs = autoencoder.predict(validation_ds_x[:10])

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(validation_ds_x[i].reshape(96, 96))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(96, 96))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
