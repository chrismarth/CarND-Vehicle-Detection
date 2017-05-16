import glob
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
from features import get_hog_features
from features import bin_spatial
from features import color_hist


def get_features(img):
    """
    Given an image return a feature vector that can be passed to a classifier
    :param img: The image from which to calculate a feature vector
    :return: Feature vector containing color histogram and HOG features on first two color channels
    """
    # Get Histogram of Color features for spatially binned image
    hist_features = color_hist(img)

    # Get HOG features for each channel in the image
    hog1 = get_hog_features(img[:, :, 0], 9, 8, 2)
    hog2 = get_hog_features(img[:, :, 1], 9, 8, 2)
    hog_features = np.hstack((hog1, hog2))

    # Combine features into a single feature vector
    return np.hstack((hist_features, hog_features))


def prepare_data(train_data_dir, return_scaler=False):
    """
    Given a directory containing training data create a scaled feature vector and labels that
    can be used in a classifier. Optionally, return the scaler that was used for the data set
    so that it can be used in the test set
    
    :param train_data_dir: The directory containing training data
    :param return_scaler: Whether or not to return the scaler
    :return: Scaled feature and label data and, optionally, the scaler used to generate the feature vector
    """
    features = []
    y = []
    for img_file in glob.glob(train_data_dir + "/**/*.png", recursive=True):
        img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2YCrCb)
        img_features = get_features(img)
        features.append(img_features)
        y.append(0 if "non-vehicles" in img_file else 1)

    X = np.vstack(features)
    X_scaler = StandardScaler().fit(X)
    X_scaled = X_scaler.transform(X)

    if return_scaler:
        return X_scaled, np.array(y), X_scaler
    else:
        return X_scaled, np.array(y)


def train_svm(X_train, y_train):
    """
    Given a feature vector and corresponding label data return an SVM classifier that can be used
    to classify images and vehicles and non-vehicles
    :param X_train: Feature vector
    :param y_train: Labels
    :return: SVM classifier 
    """
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    return clf


