import os
from PIL import Image
from numpy import asarray

def convert_image(image):
    numpy_array = asarray(image) # shape (width, height, channels)
    grey_scale = numpy_array.mean(axis=2) # shape (width, height)
    grey_image = Image.fromarray(grey_scale.astype('uint8'))
    return grey_image


def convert_directory(source, target):
    for root, dirs, files in os.walk(source):
        for fname in files:
            convert_image(Image.open(os.path.join(root, fname))).save(os.path.join(target, fname))
            