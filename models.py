"""

"""

from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Lambda, Dropout, MaxPooling2D



def lenet(input_shape, keep_prob = 0.5):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape)) # normalize data
    model.add(Convolution2D(6, 5, 5, activation = 'relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation = 'relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(keep_prob))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


def nvidia_cnn(input_shape, keep_prob = 0.5):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape)) # normalize data
    model.add(Convolution2D(24, 5, 5, activation = 'relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation = 'relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation = 'relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(keep_prob))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
    
