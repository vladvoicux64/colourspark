import keras
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from stl10_input import read_all_images

# because this model will only need unsupervised training
# we will treat the original training set as a validation set
# and use only the unlabeled set for training

TRAIN_DATA_PATH = './data/stl10_binary/unlabeled_X.bin'
VALIDATION_DATA_PATH = './data/stl10_binary/train_X.bin'
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
if __name__ == "__main__":
    # TODO: take the images, use opencv to convert to hsl, and create the tensoflow datasets


    train_images = read_all_images(TRAIN_DATA_PATH)
    validation_iamges = read_all_images(VALIDATION_DATA_PATH)
    train_ds = keras.utils.image_dataset_from_tensor_slices(train_images,
                                                         shuffle = True,
                                                         batch_size = BATCH_SIZE,
                                                         image_size = IMG_SIZE)

    validation_ds = keras.utils.image_dataset_from_tensor_slices(validation_iamges,
                                                               shuffle=True,
                                                               batch_size=BATCH_SIZE,
                                                               image_size=IMG_SIZE)