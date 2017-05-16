import cv2
import numpy as np
from scipy.ndimage.measurements import label

from features import get_hog_features
from features import bin_spatial
from features import color_hist


def get_heatmap(bbox_list, size=(720,1280)):
    """
    Given a list of bounding boxes, generate a heatmap where the value at each pixel is the number of
    overlapping bounding boxes
    
    :param bbox_list: List of bounding boxes
    :param size: Size of the heatmap to be generated
    :return: heatmap
    """
    heatmap = np.zeros(size)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def draw_labeled_bboxes(img, heatmap):
    """
    Given an image and heatmap, draw bounding boxes for each heatmap
    
    :param img: image on which to draw the bounding boxes
    :param heatmap: heatmap that bounding boxes are derived from
    :return: image with bounding boxes drawn on it
    """
    # Iterate through all detected cars
    labels = label(heatmap)
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def find_cars(img, ystart, ystop, xstart, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    """
    Given an image, perform a sliding window search to detect vehicles in the image frame
    
    :param img: The image in which to search for vehicles
    :param ystart: The minimum y-value in the image in which to start the sliding window search
    :param ystop: The maximum y-value in the image in which to finish the sliding window search
    :param xstart The minimum x-value in the image in which to start the sliding window search
    :param scale: The scale of the image we are searching for
    :param svc: The SVM classifier that is used to detect vehicles in the slideing window
    :param X_scaler: The scaler that is used to scale the feature vector passed to the SVM classifier
    :param orient: The number of HOG orientations to calculate in the feature vector
    :param pix_per_cell: The number of pixels in each detection cell
    :param cell_per_block: The number of cells in each detection block
    :return: list of bounding boxes where the classifier has detected vehicles
    """

    img_tosearch = img[ystart:ystop, xstart:, :]
    if scale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = img_tosearch[:, :, 0]
    ch2 = img_tosearch[:, :, 1]
    ch3 = img_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch2.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch2.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # List of bounding boxes where the model detects a vehicle
    bboxes = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            #hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            #spatial_features = bin_spatial(subimg)
            hist_features = color_hist(subimg)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bboxes.append(((xbox_left + xstart, ytop_draw + ystart), (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))

    return np.array(bboxes)
