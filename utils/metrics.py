import numpy as np 
import cv2 as cv
from scipy.stats import wasserstein_distance
from skimage.measure import compare_ssim
import fid
import torch

def mean_squared_error(img1, img2):
    return np.mean((img1.astype(np.float) - img2.astype(np.float)) ** 2)

def peak_signal_to_noise_ratio(real, fake):
    """
    Calculates the peak signal-to-noise ratio between two images.
    Input:
    real :: 2D NumPy array that is a real greyscale image
    fake :: 2D NumPy array that is a generated greyscale image
    pixel_range :: list of two elements representing the smallest and largest pixel intensities

    Returns:
    the peak signal-to-noise ratio between the two greyscale images
    """

    mse = mean_squared_error(real, fake)
    if mse == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(mse))

def earth_movers_distance(real, fake):
    real_histogram = cv.calcHist(real, [0], None, [256], [0, 256])
    fake_histogram = cv.calcHist(fake, [0], None, [256], [0, 256])
    return wasserstein_distance(real_histogram, fake_histogram)

def structural_similarity_index_measure(real, fake):
    return compare_ssim(real, fake)

def frechet_inception_distance(real_image_path, fake_image_path):
    return fid.compute(real_image_path, fake_image_path, gpu='cuda:0')

