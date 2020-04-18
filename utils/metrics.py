import numpy as np 
import cv2 as cv
from scipy.stats import wasserstein_distance
from skimage.measure import compare_ssim
import fid
import torch

def mean_squared_error(img1, img2):
    """
    Calculates the mean squared error between two images

    Input:
    -- img1 :: 2D NumPy array that is a greyscale image
    -- img2 :: 2D NumPy array that is a greyscale image

    Returns:
    -- :: the mean squared error between two grayscale images
    """
    return np.mean((img1.astype(np.float) - img2.astype(np.float)) ** 2)

def peak_signal_to_noise_ratio(real, fake):
    """
    Calculates the peak signal-to-noise ratio between two images.

    Input:
    -- real :: 2D NumPy array that is a real greyscale image
    -- fake :: 2D NumPy array that is a generated greyscale image

    Returns:
    -- :: the peak signal-to-noise ratio between the two greyscale images
    """

    mse = mean_squared_error(real, fake)
    if mse == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(mse))

def earth_movers_distance(real, fake):
    """
    Calculates the wasserstein distance (or earth movers distance) between two greyscale images.

    Input:
    -- real :: 2D NumPy array that is a real greyscale image
    -- fake :: 2D NumPy array that is a generated greyscale image
  
    Returns:
    -- :: the earth movers distance between the two greyscale images
    """

    real_histogram = np.zeros(256)
    fake_histogram = np.zeros(256)
    real_values, real_counts = np.unique(real, return_counts=True)
    fake_values, fake_counts = np.unique(fake, return_counts=True)
    for rv, rc in zip(real_values, real_counts):
        real_histogram[rv] = rc
    for fv, fc in zip(fake_values, fake_counts):
        fake_histogram[fv] = fc
    return wasserstein_distance(real_histogram, fake_histogram)

def structural_similarity_index_measure(real, fake):
    """
    Calculates the structural similarity index measure between two greyscale images.

    Input:
    -- real :: 2D NumPy array that is a real greyscale image
    -- fake :: 2D NumPy array that is a generated greyscale image
  
    Returns:
    -- :: the structural similarity index measure between the two greyscale images
    """

    return compare_ssim(real, fake)

def frechet_inception_distance(real_image_path, fake_image_path):
    """
    Calculates the Frechet Inception distance between two sets of greyscale images.

    Input:
    -- real_image_path :: the path to a directory of real images
    -- fake_image_path :: the path to a directory of fake images
  
    Returns:
    -- :: the Frechet Inception distance between the two sets
    """

    return fid.compute(real_image_path, fake_image_path)

if __name__ == '__main__':
    real = np.random.randint(0, 256, size=(256, 256))
    fake = np.random.randint(0, 256, size=(256, 256))

    mse = mean_squared_error(real, fake)
    psnr = peak_signal_to_noise_ratio(real, fake)
    emd = earth_movers_distance(real, fake)
    ssim = structural_similarity_index_measure(real, fake)

    ## FID might not be as helping since the images are grayscale and the CNN takes 3 channels (RGB)
    # frechet = frechet_inception_distance(real_image_path='/Users/loganfrank/Desktop/code/mlcv-final-project/data/images/test/test1', fake_image_path='/Users/loganfrank/Desktop/code/mlcv-final-project/data/images/test/test2')

    print(f'MSE: {mse}')
    print(f'PSNR: {psnr}')
    print(f'EMD: {emd}')
    print(f'SSIM: {ssim}')
    # print(f'FID: {frechet}')
