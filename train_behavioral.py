import os
import os.path as osp
import sys
import argparse
import numpy as np
import cv2
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from models import lenet, nvidia_cnn

def load_data(fn):
    """
    Reads the csv with image location and steering angle 
    """
    images = []
    steering_angles = []
    offset = 0.275 # offset for right and left 
    with open(fn) as f:
        reader = csv.reader(f)
        next(reader, None)
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            images.append([center_img.strip(), left_img.strip(), right_img.strip()])
            steering_angles.append([angle, angle+offset, angle-offset])
            
    return images, steering_angles

def get_image(img_data):
    # Read image from path
    img = cv2.imread(osp.join('data', img_data[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_img(img)
    img = resize(img)
    return np.array(img, dtype=np.float32)

def resize(img):
    # Resize the image to fit the network input shape
    return cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)

def crop_img(img):
    # Crops image to show only ROI
    return img[40:-20, :]

def batch_generator(X_data, labels, batch_size = 32):
    num_images = len(X_data)
    total_batch = int(num_images / batch_size)
    indeces = [ix for ix in range(num_images)]
    while 1:
        shuffle(indeces)
        i = 0
        for _ in range(total_batch):
            X_batch = []
            y_batch = []
            for j in range(batch_size):
                X_batch.append(get_image(X_data[indeces[i]]))
                y_batch.append(labels[indeces[i]])
                i = i + 1
            print(y_batch)
            yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

def train_model(model, X_train, X_validation, y_train, y_validation, batch_size = 32, epochs = 100, learning_reate = 0.001):
    early_stopping_monitor = EarlyStopping(patience=15) # if not improved over 10 epochs, stop training
    model.fit_generator(batch_generator(X_train, y_train, batch_size = batch_size), samples_per_epoch=24000, nb_epoch=epochs, verbose=1, callbacks=[early_stopping_monitor], validation_data=batch_generator(X_validation, y_validation, batch_size), nb_val_samples=1024)
    model.save(osp.join('models', 'model_{}_{}.hdf5'))

def parse_args():
    parser = argparse.ArgumentParser(description='Train the behavioral model')
    parser.add_argument('--batch', dest='batch',
                        help='batch size to train',
                        default=64, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to run over training set',
                        default=100, type=int)
    parser.add_argument('--l', dest='learning_rate',
                        help='set the learning rate of model',
                        default=0.001, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
    Train behavioral model
    """
    args = parse_args()
    X, y = load_data(osp.join('data', 'driving_log.csv'))
    X, y = shuffle(X, y)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=14)
    model = nvidia_cnn(input_shape = (66, 200, 3))
    model.summary() # print summary to debug model
    train_model(model, X_train, X_validation, y_train, y_validation)
    
