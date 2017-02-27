import matplotlib.image as mpimg
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import time

from image_feature_extractor import ImageFeatureExtractor
import config


if __name__ == '__main__':
    print('Extracting features...')
    t = time.time()

    ife = ImageFeatureExtractor(config.feature_config)

    # Divide up into cars and notcars
    cars = glob.glob('data/vehicles/*/*.png')
    notcars = glob.glob('data/non-vehicles/*/*.png')

    car_features = []
    notcar_features = []

    for file in cars:
        image = mpimg.imread(file)
        car_feature = ife.extract_features(image)
        car_features.append(car_feature)

    for file in notcars:
        image = mpimg.imread(file)
        notcar_feature = ife.extract_features(image)
        notcar_features.append(notcar_feature)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract features...')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', config.feature_config['orient'], 'orientations', config.feature_config['pix_per_cell'],
          'pixels per cell and', config.feature_config['cell_per_block'], 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    print('\nTraining classifier...')
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Pickle results
    with open('./trained_model/model.pkl', 'wb') as f:
        pickle.dump(svc, f)

    with open('./trained_model/xscaler.pkl', 'wb') as f:
        pickle.dump(X_scaler, f)