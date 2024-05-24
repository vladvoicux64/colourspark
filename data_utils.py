import cv2
import numpy as np


def convert_to_grayscale(images):
    output = []
    for image in images:
        output.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    return np.array(output).reshape((len(images), 96, 96, 1))
