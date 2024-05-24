import keras
from stl10_input import read_all_images
from data_utils import convert_to_hls
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf


# because this model will only need unsupervised training
# we will treat the original training set as a validation set
# and use only the unlabeled set for training

TRAIN_DATA_PATH = './data/stl10_binary/unlabeled_X.bin'
VALIDATION_DATA_PATH = './data/stl10_binary/train_X.bin'
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
if __name__ == "__main__":
    # TODO: take the images, use opencv to convert to hsl, and create the tensoflow datasets

    # read images from disk
    train_images = read_all_images(TRAIN_DATA_PATH)

    validation_images = read_all_images(VALIDATION_DATA_PATH)

    # convert to hsl
    train_images = convert_to_hls(train_images)

    validation_images = convert_to_hls(validation_images)

    # create datasets
    train_ds = tf.data.Dataset.from_tensor_slices(train_images)

    validation_ds = tf.data.Dataset.from_tensor_slices(validation_images)
