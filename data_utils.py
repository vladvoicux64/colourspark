import numpy as np
import cv2


def convert_to_hls(images):
    RGB2HLS_conversion = np.vectorize(lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2HLS, image))
    return RGB2HLS_conversion(images)
