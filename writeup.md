# Vehicle Detection Project

The goals of this project were the following:

* Perform Histogram of Oriented Gradients (HOG) and histogram of color feature extraction on a labeled training set of vehicle and non-vehicle images in order to train a Linear SVM classifier.
* Apply this classifier within a video processing pipeline in order to identify vehicles by estimating and drawing a bounding box around each vehicle in each video frame.

[//]: # (Image References)
[image1]: ./examples/features.png
[image2]: ./examples/heatmap.png
[image3]: ./examples/sliding_window.png

### Classifier Development 

The classifier created for this project was developed from vehicle and non-vehicle GTI and KTTI image data provided with the project available [(vehicles)](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [(non-vehicles)](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

#### 1. Histogram of Oriented Gradients (HOG)

The code used to extract features, including HOG features, from the test images is contained within the file `features.py`. With respect to the HOG features used, many different setups, including converting to different color spaces and the number of channels to include, were tried before finally settling on transforming the images into the YCrCb colorspace and extracting HOG features for the Y and Cr channels.  

Different HOG parameters were also tried, with the following table summarizing the values that were ultimately used during classification:

| Parameter  | Value|
|:----------:|:----:|
|Orientations|  9   |
|Pixels/Cell | (8,8)|
|Cells/Block | (2,2)|


An image of the Histogram of Color and HOG for the Cr channel for a sample of images from the training set is shown in the following figure

![alt text][image1]


#### 2. Color Features 

While it was possible to produce a classifier that was able to distinguish vehicles from non-vehicles to an accuracy of roughly 94%, vehicle detection performance on the test video was inadequate. There were too many false positives and the bounding boxes around vehicles were too large. In order to improve performance, a histrogram of all of the YCrCb channels was also included in the feature vector. While this only slightly improved classifier performance, vehicle detection in the test video was greatly improved and produced acceptable results. 

#### 3. Classifier training

Due to it's ability to separate values in a complicated domain with clear separation (The histogram of color and HOG features show that the vehicle and non-vehicle images have clear differences) and the fact that the training data set was not too large, a linear support vector machine, as provided from the ski-kit learn svm package, was used as the classifier for this project. The code to train the SVM is contained both in the python file `svm_train.py` and in the Jupyter notebook `run_model` under the heading **Setup and Train SVM**. The default parameters for kernel ('rbf'), C-value (1.0), and Gamma ('auto') were used as they provided adequate classifier performance.

### Video Processing 

With the classifier trained, focus shifted to using it to identify vehicles within the test video. This was primarily done using a sliding window approach. That is, within each image frame of the video we search for vehicles within a smaller subset of the image - the window. We then slide this window across the image space using the classifier to look for vehicles within the window.

#### 1. Sliding Window Search

Instead of searching the entire image space with the sliding window, the image was only searched below the horizon and to the right of the center-lane. Limiting the search space has several benefits. It reduces the number of false-positives from oncoming traffic and unimportant image features on the opposite side of the road. It also greatly improves performance since we avoid unnecessary searching where we know vehicles would not or should not be detected. 

Pipeline performance was improved by calculating the HOG for the entire region of interest once. The HOG calculation seemed to be the most expensive operation when processing each image frame, and by calculating it just once image processing performance was greatly improved. Once the HOG features for the entire image were calculated we were able to grab the appropriate subsections based on the spatial location of the current window being searched.

The size of the search window was also adjusted across the image space. As should be obvious, vehicles that are farther away from the camera appear smaller and vehicles that are closer appear larger. Therefore, if we are to give our classifier the best possible chance to identify vehicles accurately, we need to adjust the size of search window when searching different parts of the image. To this end, the search was performed using four different search scales within four different regions. These scales and image subsets were determined while experimenting and provided the best results while also maintaining reasonable performance (if the scale was made too small performance greatly suffered). A summary of the scales and search regions are given in the following table:

| Scale  | Y-Region Bounds|
|:------:|:--------------:|
|1.0     |   (400, 500)   |
|1.5     |   (400, 550)   |
|2.0     |   (400, 650)   |
|3.0     |   (550, 720)   |

With this configuration, the sliding window search was able to accurately detect vehicle without much difficulty. The code that performs the sliding window search is contained in the function `find_cars` within the python file `window_search.py` and an example of how this part of the pipeline was run is contained within the Jupyter notebook `run_model.ipynb` under the section title **Sliding Window Search**. The following set of images show the bounding boxes that were generated while performing the vehicle search:

![alt text][image3]

#### 2. Drawing the Vehicle Bounding Box

If the sliding window finds a match for a vehicle, a bounding box is drawn around the vehicle. The entire set of bounding boxes found in a single frame are then sent to a heatmap function in an attempt to consolidate multiple bounding boxes into a single bounding box that encapsulates the entire vehicle. This function takes all bounding boxes and adds "heat" to an image. If multiple bounding boxes overlap, the image gets "hotter" where the boxes overlap. Finally, the fully constructed heatmap is passed to a function that draws the final detection frames including the composite bounding box around each detected vehicle. This function uses the scipy.labels function to consolidate regions of the heatmap into logical groups (each group is given a label). A bounding box is then drawn around each of these groups to produce the final bounding box for a detected vehicle.  The code that performs the heatmap calculation and bounding box generation is contained in the functions `get_heatmap` and `draw_labeled_bboxes` within the python file `window_search.py` and an example of how this part of the pipeline was run is contained within the Jupyter notebook `run_model.ipynb` under the sections titled **Show Heatmap** and **Draw Final Image Detection Frame**. 

#### 2. Buffering frames and Rejecting False-Positives
Despite our best efforts to produce a good classifier and sliding window algorithm false-positives were unavoidable. Beyond restricting the search region, a buffering and thresholding algorithm was also applied. In essence, instead of creating the heatmap that is fed into the composite bounding box function from scratch on each frame, we take the heatmaps from a configurable number of the previous frames and combine them to create a combined heatmap. We then apply a threshold to the heatmap to filter out regions with lower heat with the assumption that regions of low heat are most likely false-positives. The code that performs this part of the processing pipeline is contained within the Jupyter notebook `run_model.ipynb` under the sections titled **Build the Complete Vehicle Detection Pipeline**. An example of the heatmap and composite bounding boxes that result for the same sample images above are shown in the folowing image:

![alt text][image2]
---

### Video Implementation

All of the steps of the processing pipeline were combined to produce the following video where vehicles were successfully detected: [Final video result](./output_video.mp4)


### Discussion

While the processing pipeline, in general, successfully identified vehicles, it had some obvious short-comings. First, performance was not on the level that it could be used in a real-time processing environment. While my personal laptop on which the pipeline was developed is a mediocre performer at best, the pipeline would most likely still need to be improved by a factor of two. This could possibly be achieved by reducing the size of the feature vector. HOG calculations are slow, and while we definitely needed a HOG feature, it's necessarily clear whether we absolutely needed the HOG for both channels that were included. The other major performance bottleneck was the sliding window search. If too small of a scale was used the number of steps in the sliding window search became large and negatively impacted performance. Again, while some effort was made to optimize the scales and search space for the sliding window algorithm, more development here could probably greatly improve performance. Another obvious deficiency in the algorithm was it's inability to distinguish two vehicles that either overlapped or were very close to each other. In these instances the algorithm would produce a large bounding box enclosing both vehicles instead of two overlapping bounding boxes. While this may ultimately be sufficient within the context of how the vehicle detections would be used, it does not accurately reflect the reality of the scene. Finally, one last deficiency was the inability of the algorithm to detect vehicles at great distance. Despite efforts to perform a sliding window search with very small scales, we were unable to pick up vehicles that were close to the horizon. And, as mentioned above, the small scales had a large negative impact on overall performance. Again, given the context, this may be acceptable even though it does not accurately reflect reality.

