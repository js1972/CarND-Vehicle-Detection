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

from object_detector import ObjectDetector
import config


if __name__ == '__main__':
    print('Processing video frames')

    with open('./trained_model/model.pkl', 'rb') as f:
        svc = pickle.load(f)
    with open('./trained_model/xscaler.pkl', 'rb') as f:
        X_scaler = pickle.load(f)

    detector = ObjectDetector(config.feature_config, config.sliding_windows_config, svc, X_scaler)

    video_output = 'output_images/project_video_output.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    #clip1 = VideoFileClip('test_video.mp4')
    clip1_output = clip1.fl_image(detector.process) #NOTE: this function expects color images!!
    clip1_output.write_videofile(video_output, audio=False)

    print('Video processing completed')