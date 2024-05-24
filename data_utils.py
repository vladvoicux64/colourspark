import cv2


def convert_to_hls(images):
    conversion = lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    for image in images:
        conversion(image)
    return images
