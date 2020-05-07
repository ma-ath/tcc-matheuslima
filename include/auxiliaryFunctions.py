import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import scipy.io.wavfile
import matplotlib.pyplot as plt
from math import floor, log
import re
import pickle
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import model_from_json
from network_model import networkModel
import numpy as np
from PIL import Image
import re
from matplotlib import pyplot as plt
import pandas
from random import randrange, seed

def loadDataset(PROCESSED_DATA_FOLDER,image_shape):
    training_images = np.load(PROCESSED_DATA_FOLDER+"images_training-img.npy")
    training_labels = np.load(PROCESSED_DATA_FOLDER+"images_training-lbl.npy")
    testing_images = np.load(PROCESSED_DATA_FOLDER+"images_testing-img.npy")
    testing_labels = np.load(PROCESSED_DATA_FOLDER+"images_testing-lbl.npy")

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for image in training_images:
        X_train.append(image.reshape(image_shape))
    for label in training_labels:
        Y_train.append(label)
    for image in testing_images:
        X_test.append(image.reshape(image_shape))
    for label in testing_labels:
        Y_test.append(label)

    #Transform the loaded data to numpy arrays
    X_train = np.array(X_train).astype("float32")
    Y_train = np.array(Y_train).astype("float32")
    X_test = np.array(X_test).astype("float32")
    Y_test = np.array(Y_test).astype("float32")

    #Normalize the input image from 0-255 to 0-1
    #I do it before loading the dataset because if done before, i would have to save the
    #training file as float32, which takes considerably more space than a uint8 format

    X_train = X_train / 255
    X_test = X_test / 255

    return X_train,Y_train,X_test,Y_test

def loadDataset_testOnly(PROCESSED_DATA_FOLDER,image_shape):
    testing_images = np.load(PROCESSED_DATA_FOLDER+"images_testing-img.npy")
    testing_labels = np.load(PROCESSED_DATA_FOLDER+"images_testing-lbl.npy")

    X_test = []
    Y_test = []

    for image in testing_images:
        X_test.append(image.reshape(image_shape))
    for label in testing_labels:
        Y_test.append(label)

    #Transform the loaded data to numpy arrays
    X_test = np.array(X_test).astype("float32")
    Y_test = np.array(Y_test).astype("float32")

    #Normalize the input image from 0-255 to 0-1
    #I do it before loading the dataset because if done before, i would have to save the
    #training file as float32, which takes considerably more space than a uint8 format

    X_test = X_test / 255

    return X_test,Y_test

def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)
    #model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)

def save_weights(model):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)

def load_model():
    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())
    model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model

#Function that returns if a program is installed at the machine
def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

#Function that plots both power and time series of an audio at once
def plotAudio(FSample,samples,M,St):
    plt.figure("Audio Information")

    plt.subplot(211)
    audio_length1 = samples.shape[0] / FSample
    time1 = np.linspace(0., audio_length1, samples.shape[0])
    plt.plot(time1, samples[:, 0], label="Left channel")
    plt.plot(time1, samples[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio timeline")

    plt.subplot(212)
    audio_length2 = (M*St[:,0].size)/FSample
    time2 = np.linspace(0., audio_length2, St[:,0].size)
    plt.plot(time2, St[:, 0], label="Left channel")
    plt.plot(time2, St[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Power stream timeline")

    plt.show()
    pass

