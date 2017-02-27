import numpy as np
import cv2
from scipy.ndimage.measurements import label

from image_feature_extractor import ImageFeatureExtractor


class ObjectDetector(object):
    """
    Detect objects in an image using the provided image
    feature extractor and a trained Linear SVM classifier.
    """

    def __init__(self, config, sliding_window_config, classifier, standard_scaler):
        """
        """
        self.config = config
        self.sliding_window_config = sliding_window_config
        self.classifier = classifier
        self.standard_scaler = standard_scaler
        # self.heatmap = []
        # self.windows = []

    def process(self, image):
        """
        """
        global heat
        global windows

        draw_image = np.copy(image)
        original_image = np.copy(image)

        if np.max(image) > 1:
            image = image.astype(np.float32) / float(np.max(image))

        windows = self.get_pyramid_windows(image)
        positive_detections = self.find_objects(image, windows)
        # window_img = self.draw_boxes(draw_image, positive_detections)
        # new heatmap
        heat = np.zeros_like(draw_image[:, :, 0]).astype(np.float)
        heatmap = self.add_heat(heat, positive_detections)
        heatmap = self.threshold_heatmap(heatmap, self.sliding_window_config['heat_threshold'])
        # Blob extraction
        # Implements: https://en.wikipedia.org/wiki/Connected-component_labeling
        labels = label(heatmap)
        output_image = self.draw_labeled_bboxes(original_image, labels)
        return output_image

    def find_objects(self, img, windows):
        """
        Loop though all the provided windows and resize the part of the image occupied
        by the window to 64 x 64 to match the size of our test data image.
        With each resized part of the image: extract features and classify.
        """
        # Define a function you will pass an image
        # and the list of windows to be searched (output of slide_windows())
        found_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = ImageFeatureExtractor(self.config).extract_features(test_img)
            test_features = self.standard_scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.classifier.predict(test_features)
            if prediction == 1:
                found_windows.append(window)

        return found_windows

    def get_pyramid_windows(self, image):
        """
        Repeatedly get sliding windows for different window sizes.
        This will allow our detector to find smaller and larger sized
        objects.
        """
        windows = []
        for xy in self.sliding_window_config['window_sizes']:
            window = self.get_sliding_windows(image, xy_window=(xy, xy))
            windows += window
        return windows

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def get_sliding_windows(self, img, xy_window):
        """
        Implementation of the sliding-window technique.
        Return a list of windows (rectangles) that have been moved over
        the image with the given size and configured overlap and start/stop
        positions.
        """
        # If x and/or y start/stop positions not defined, set to image size
        if self.sliding_window_config['x_start_stop'][0] == None:
            self.sliding_window_config['x_start_stop'][0] = 0
        if self.sliding_window_config['x_start_stop'][1] == None:
            self.sliding_window_config['x_start_stop'][1] = img.shape[1]
        if self.sliding_window_config['y_start_stop'][0] == None:
            self.sliding_window_config['y_start_stop'][0] = 0
        if self.sliding_window_config['y_start_stop'][1] == None:
            self.sliding_window_config['y_start_stop'][1] = img.shape[0]

        xspan = self.sliding_window_config['x_start_stop'][1] - self.sliding_window_config['x_start_stop'][0]
        yspan = self.sliding_window_config['y_start_stop'][1] - self.sliding_window_config['y_start_stop'][0]
        nx_pix_per_step = np.int(xy_window[0] * (1 - self.sliding_window_config['xy_overlap'][0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - self.sliding_window_config['xy_overlap'][1]))
        nx_windows = np.int(xspan / nx_pix_per_step) - 1
        ny_windows = np.int(yspan / ny_pix_per_step) - 1
        window_list = []
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + self.sliding_window_config['x_start_stop'][0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + self.sliding_window_config['y_start_stop'][0]
                endy = starty + xy_window[1]
                window_list.append(((startx, starty), (endx, endy)))
        return window_list

    def draw_boxes(self, img, bboxes, color=(0, 25, 255), thick=6):
        """
        Draw the given bounding boxes on the image
        """
        imcopy = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy

    def add_heat(self, heatmap, bbox_list):
        """
        Build up a heat map image from the given bounding
        boxes. We add +1 for each pixel within the bounding
        box.
        """
        for box in bbox_list:
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap

    def threshold_heatmap(self, heatmap, threshold=1):
        """
        Zero out pixels in the provided image below the given threshold

        :param heatmap: Heatmap array - same size as image
        :param threshold: The threhold value below which pixes are zeroed
        :return: Extracted features as one long feature vector
        """
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        """
        """
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 25, 255), 6)
        # Return the image
        return img