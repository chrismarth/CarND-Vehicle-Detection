import cv2
import numpy as np
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False):
    """

    :param img: 
    :param orient: 
    :param pix_per_cell: 
    :param cell_per_block: 
    :param vis: 
    :return: 
    """

    if vis:
        return hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   visualise=True)
    else:
        return hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   visualise=False)


def bin_spatial(img, size=(32, 32)):
    """

    :param img: 
    :param size: 
    :return: 
    """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    """

    :param img: 
    :param nbins: 
    :return: 
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
