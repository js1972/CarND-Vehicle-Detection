import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import pickle
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import time

from scipy.ndimage.measurements import label

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


class ImageFeatureExtractor(object):
    """
    Extract features from images using:
     - Histogram of Oriented Gradients (HOG)
     - Spatial Binning
     - Colour Histogram
    """

    def __init__(self, config):
        self.config = config

    def extract_features(self, image):
        """
        Extract features from an image.

        :param image: Image array (e.g. mpimg.imread())
        :return: Extracted features as one long feature vector
        """
        # Create a list to append feature vectors to
        features = []

        if np.max(image) > 1:
            image = image.astype(np.float32) / float(np.max(image))

        # apply color conversion if other than 'RGB'
        if self.config['color_space'] != 'RGB':
            if self.config['color_space'] == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.config['color_space'] == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif self.config['color_space'] == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.config['color_space'] == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif self.config['color_space'] == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if self.config['spatial_feat'] == True:
            spatial_features = self.bin_spatial(feature_image)
            features.append(spatial_features)

        if self.config['hist_feat'] == True:
            # Apply color_hist()
            hist_features = self.color_hist(feature_image)
            features.append(hist_features)

        if self.config['hog_feat'] == True:
            if self.config['hog_channel'] == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_feature = self.get_hog_features(feature_image[:, :, channel])
                    hog_features.append(hog_feature)

                hog_features = np.ravel(hog_features)
            else:
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel])

            # Append the new feature vector to the features list
            features.append(hog_features)

        # Return list of feature vectors
        return np.concatenate(features)

    def bin_spatial(self, img):
        size = self.config['spatial_size']
        color1 = cv2.resize(img[:, :, 0], size).ravel()
        color2 = cv2.resize(img[:, :, 1], size).ravel()
        color3 = cv2.resize(img[:, :, 2], size).ravel()
        return np.hstack((color1, color2, color3))

    def color_hist(self, img):
        nbins = self.config['hist_bins']  # bins_range=(0, 256)
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def get_hog_features(self, image):
        features = hog(image,
                       orientations=self.config['orient'],
                       pixels_per_cell=(self.config['pix_per_cell'], self.config['pix_per_cell']),
                       cells_per_block=(self.config['cell_per_block'], self.config['cell_per_block']),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=True)

        return features