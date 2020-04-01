## Basic Python imports
import math
import random 

## PyTorch imports
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF 

## Image / Array imports
from PIL import Image
from PIL import ImageFilter
import numpy as np 

class RotationTransform(object):
    """
    Rotates a PIL image by 0, 90, 180, or 270 degrees. Randomly chosen using a uniform distribution.
    """
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, image):
        angle = random.choice(self.angles)
        return TF.rotate(image, angle)
    
    def get_params(self):
        return random.choice(self.angles)


class GammaJitter(object):
    """
    Jitters the gamma of a PIL image between a uniform distribution of two values (low & high).
    Larger gammas make the shadows darker, smaller gammas make the shadows lighter.
    """
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high
    
    def __call__(self, image):
        gamma = np.random.uniform(self.low, self.high)
        return TF.adjust_gamma(image, gamma)
    
    def get_params(self):
        return np.random.uniform(self.low, self.high)

class BrightnessJitter(object):
    """
    Jitters the gamma of a PIL image between a uniform distribution of two values (low & high).
    Larger gammas make the shadows darker, smaller gammas make the shadows lighter.
    """
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high
    
    def __call__(self, image):
        factor = np.random.uniform(self.low, self.high)
        return TF.adjust_brightness(image, factor)

class RandomScale(object):
    """
    Scales a PIL image based on a value chosen from a uniform distribution of two values (low & high).
    """
    def __init__(self, low=1.0, high=1.1):
        self.low = low
        self.high = high

    def __call__(self, image):
        height = image.height
        width = image.width
        scale = np.random.uniform(self.low, self.high)
        image = TF.resize(image, (math.floor(height * scale), math.floor(width * scale)))
        return TF.center_crop(image, (height, width))

class Resize(object):
    """
    Scales a PIL image based on a value chosen from a uniform distribution of two values (low & high).
    """
    def __init__(self, resolution=1, size=0):
        self.resolution = resolution
        self.size = size

    def __call__(self, image):
        if self.size == 0:
            height = image.height
            width = image.width
            return image.resize((width // self.resolution, height // self.resolution))
        else:
            return image.resize((self.size, self.size), Image.BILINEAR)

class AdaptiveCenterCrop(object):
    """
    Center crops the image to be a square of the smallest edge squared.
    """
    def __init__(self):
        pass

    def __call__(self, image):
        length = min(image.width, image.height)
        return TF.center_crop(image, (length, length))

class MedianFilter(object):
    """
    Randomly applies a median filter to an image.
    """
    def __init__(self, filter_size=3, p=0.1):
        self.filter_size = filter_size
        self.p = p

    def __call__(self, image):
        roll = random.random()
        if roll < self.p:
            return image.filter(ImageFilter.MedianFilter(self.filter_size))
        else:
            return image

class RGBNIRTransform(object):
    def __init__(self, resize=None, hflip=None, vflip=None, rotation=None, gamma_jitter=None, normalize=None, train=True):
        self.resize = Resize(size=resize)
        if train:
            self.hflip = hflip
            self.vflip = vflip
            self.rotation = RotationTransform()
            self.gamma_jitter = GammaJitter(low=gamma_jitter[0], high=gamma_jitter[1])
        self.normalize = transforms.Normalize(mean=normalize[0], std=normalize[1])
        self.nir_normalize = transforms.Normalize(mean=[0.5], std=[0.25])
        self.train = train

    def __call__(self, rgb, nir):
        # Resize both images
        rgb = self.resize(rgb)
        nir = self.resize(nir)

        # Perform the training transforms
        if self.train:
            # Check for random horizontal flip
            num = random.random()
            if num < self.hflip:
                rgb = TF.hflip(rgb)
                nir = TF.hflip(nir)
            
            # Check for random vertical flip
            num = random.random()
            if num < self.vflip:
                rgb = TF.vflip(rgb)
                nir = TF.vflip(nir)

            # Random rotation
            rotation_angle = self.rotation.get_params()
            rgb = TF.rotate(rgb, rotation_angle)
            nir = TF.rotate(nir, rotation_angle)

            # Gamma adjustment
            gamma_adjustment = self.gamma_jitter.get_params()
            rgb = TF.adjust_gamma(rgb, gamma_adjustment)
            nir = TF.adjust_gamma(nir, gamma_adjustment)
        
        # Convert to PyTorch tensor
        rgb = TF.to_tensor(rgb)
        nir = TF.to_tensor(nir)

        # Normalize both images
        rgb = self.normalize(rgb)
        nir = self.nir_normalize(nir)

        return rgb, nir

    def __repr__(self):
        pass
