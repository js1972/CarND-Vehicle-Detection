# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---
Notes:

The accompanying Jupyter Notebook `Vehicle Detection.ipynb` contains the code for this project and is used for testing various configuration options.
The classes `ImageFeatureExtractor` and `ObjectDetector` have additionally been copied into their own python files so that they can be used elsewhere.
The python scripts `train.py` and `process.py` uses these classes to perform the training and pipeline processing outside of the notebook as I was getting out-of-memory issues processing the larger video files in Jupyter.

## Feature Extraction - Histogram of Oriented Gradients (HOG)
In cell 2 of the notebook I have defined the `ImageFeatureExtractor` class. This class is instantiated with a list of configuration options for feature extraction.
Method `extract_features` does the hard work and performs the following steps:
- Colour-space conversion. Allows for conversion to one of the OpenCV supported colour spaces.
- Extract spatial features. Extract features by collecting colour spatial information (per channel) and concatenating to form a feature vector.
- Extract colour histograms. Using `np.histogram` we compute histograms for each image channel and then concatenate them together to form the features.
- Extract HOG features. The `skimage.feature.hog` function is used here to extract features per image channel. Based on the provided config (setup in cell 3). It allows us to extract features on an individual channel of choice or across all channels.

The results of each of the above feature extraction techniques is then concatenated together to form one long feature vector at the end of method `extract_features`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

The main technique here is extracting the HOG features. I settled on using the other colour techniques as well but found that they made little difference to the output. Possibly tweaking the parameters here could have achieved a better result. Dealing with blob detection false positives was much more important to the final pipeline and these are discussed further below.

My final configuration options are as follows:
```
feature_config = {
    'color_space' : 'YCrCb',
    'orient' : 9,
    'pix_per_cell' : 8,
    'cell_per_block' : 2,
    'hog_channel' : 'ALL',
    'spatial_size' : (32, 32),
    'hist_bins' : 32,
    'spatial_feat' : True,
    'hist_feat' : True,
    'hog_feat' : True
}
```

Example HOG features:
![alt text](output_images/HOG features.png)



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
