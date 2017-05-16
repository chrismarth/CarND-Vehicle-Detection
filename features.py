import cv2
import numpy as np
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True, vis=False):
    """
    Given an image, calculate HOG features

    :param img: The image to calculate HOG features
    :param orient: The number of HOG orientations to calculate in the feature vector
    :param pix_per_cell: The number of pixels in each detection cell
    :param cell_per_block: The number of cells in each detection block
    :param feature_vec: Whether or not to return a feature vector
    :param vis: Whether or not to return a HOG visualization
    :return: Feature vector and optionally a HOG visualization
    """
    if vis:
        return hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   feature_vector=feature_vec,
                   visualise=True)
    else:
        return hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   feature_vector=feature_vec,
                   visualise=False)


def bin_spatial(img, size=(32, 32)):
    """
    Given an input image perform a spatial binning
    
    :param img: image from which to perform the spatial binning
    :param size: the size of the spatial binning
    :return: spatial binning of given size
    """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    """
    Given an input image return color histogram of with the given number of bins
    
    :param img: The image to retrieve the histogram from
    :param nbins: The number of color histogram bins
    :return: Histograms of color with the given number of bins
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
