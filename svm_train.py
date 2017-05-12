import glob
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
from features import get_hog_features
from features import bin_spatial
from features import color_hist


def get_features(img):
    # Get Histogram of Color features for spatially binned image
    spatial_features = bin_spatial(img)
    hist_features = color_hist(img)

    # Get HOG features for each channel in the image
    hog1 = get_hog_features(img[:, :, 0], 9, 8, 2)
    hog2 = get_hog_features(img[:, :, 1], 9, 8, 2)
    hog3 = get_hog_features(img[:, :, 2], 9, 8, 2)
    hog_features = np.hstack((hog1, hog2, hog3))

    # Combine features into a single feature vector
    return np.hstack((spatial_features, hist_features, hog_features))


def prepare_data(train_data_dir, return_scaler=False):
    features = []
    y = []
    for img_file in glob.glob(train_data_dir + "/**/*.png", recursive=True):
        img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
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


def train_svm(X_train, y_train, pickel_model=False, pickel_name="model"):
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    return clf


