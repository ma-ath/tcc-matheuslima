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
import numpy as np
from PIL import Image
import re
from matplotlib import pyplot as plt
import pandas
from random import randrange, seed
from keras.applications.vgg16 import preprocess_input

def preprocess_image(image_array, usecache=False,train_or_test='train'):
    # We only need to calculate those values if we are dealing with a new dataset. So to speedup things, we can use the precalculated mean and std for train and test datasets
    if not usecache:
        mean = np.mean(image_array,axis=(0,1,2))
        std = np.std(image_array,axis=(0,1,2))
    elif train_or_test == 'train':
        mean = [3.883008, 3.883008, 3.883008]
        std = [31.528559, 31.528559, 31.528559]
    elif train_or_test == 'test':
        mean = [4.2647796, 4.2647796, 4.2647796]
        std = [33.04215, 33.04215, 33.04215]
    else:
        raise ValueError

    image_array[:, :, :, 0] -= mean[0]
    image_array[:, :, :, 1] -= mean[1]
    image_array[:, :, :, 2] -= mean[2]

    image_array[:, :, :, 0] /= std[0]
    image_array[:, :, :, 1] /= std[1]
    image_array[:, :, :, 2] /= std[2]

    mean2 = np.mean(image_array,axis=(0,1,2))
    std2 = np.std(image_array,axis=(0,1,2))

    print("Mean before preprocessin: "+str(mean))
    print("Standart deviation before preprocessing: "+str(std))
    print("Mean after preprocessin: "+str(mean2))
    print("Standart deviation after preprocessing: "+str(std2))

    return image_array

def preprocess_labels(label_array):
    mean = np.mean(label_array,axis=0)
    std = np.std(label_array,axis=0)

    label_array[:] -= mean
    
    label_array[:] /= std
    
    mean2 = np.mean(label_array,axis=0)
    std2 = np.std(label_array,axis=0)

    print("Mean before preprocessin: "+str(mean))
    print("Standart deviation before preprocessing: "+str(std))
    print("Mean after preprocessin: "+str(mean2))
    print("Standart deviation after preprocessing: "+str(std2))

    return label_array, mean, std

def loadDataset(PROCESSED_DATA_FOLDER,image_shape, timeSteps=100,lstm=False):
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

    #Normalize the input image to have "0" mean and standart deviation of "1"
    # I tried to do it using the builtin keras function "preprocess_input()", but doing it by hand has better results
    # I do it before loading the dataset because if done before, i would have to save the
    # training file as float32, which takes considerably more space than a uint8 format

    X_train = preprocess_image(X_train,usecache=True)
    X_test = preprocess_image(X_test,usecache=True)

    #This is a temporary solution, i just delete one of the audio sources
    Y_train = np.delete(Y_train, -1, axis=1)
    Y_test = np.delete(Y_test, -1, axis=1)

    if lstm == False:
        return X_train,Y_train,X_test,Y_test

    number_images = X_train.shape[0]    #total number of images on dataset
    samples = floor(number_images / timeSteps)  #number of samples for the given number of timeSteps
    throw_away_images = number_images - samples*timeSteps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

    X_new_shape = (samples,timeSteps,)+image_shape    #new input format
    Y_new_shape = (samples,timeSteps)               #new output format

    X_train = np.delete(X_train,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor
    Y_train = np.delete(Y_train,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor

    X_train = np.reshape(X_train,X_new_shape)   #reshape the tensor to include the time dimension
    Y_train = np.reshape(Y_train,Y_new_shape)   #reshape the tensor to include the time dimension

    #  ---------------------------  Do same thing to test dataset   -----------------------------#
    number_images = X_test.shape[0]    #total number of images on dataset
    samples = floor(number_images / timeSteps)  #number of samples for the given number of timeSteps
    throw_away_images = number_images - samples*timeSteps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

    X_new_shape = (samples,timeSteps,)+image_shape    #new input format
    Y_new_shape = (samples,timeSteps)               #new output format

    X_test = np.delete(X_test,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor
    Y_test = np.delete(Y_test,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor

    X_test = np.reshape(X_test,X_new_shape)   #reshape the tensor to include the time dimension
    Y_test = np.reshape(Y_test,Y_new_shape)   #reshape the tensor to include the time dimension

    return X_train,Y_train,X_test,Y_test

def loadDataset_testOnly(PROCESSED_DATA_FOLDER,image_shape,timeSteps=100,lstm=False):
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

    #Normalize the input image for the vgg16 input
    #I do it before loading the dataset because if done before, i would have to save the
    #training file as float32, which takes considerably more space than a uint8 format

    X_test = preprocess_image(X_test,usecache=True)

    #Delete one of the audios channel
    Y_test = np.delete(Y_test, -1, axis=1)

    if lstm == False:
        return X_train,Y_train,X_test,Y_test
    
    #  ---------------------------  Do same thing to test dataset   -----------------------------#
    number_images = X_test.shape[0]    #total number of images on dataset
    samples = floor(number_images / timeSteps)  #number of samples for the given number of timeSteps
    throw_away_images = number_images - samples*timeSteps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

    X_new_shape = (samples,timeSteps,)+image_shape    #new input format
    Y_new_shape = (samples,timeSteps)               #new output format

    X_test = np.delete(X_test,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor
    Y_test = np.delete(Y_test,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor

    X_test = np.reshape(X_test,X_new_shape)   #reshape the tensor to include the time dimension
    Y_test = np.reshape(Y_test,Y_new_shape)   #reshape the tensor to include the time dimension

    return X_test,Y_test

def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)

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

def plotAudioPowerWithPrediction(testSamples,predictedSamples):
    plt.figure("Audio Power")

    audio_length = testSamples.shape[0]
    time = np.linspace(0., 1/audio_length ,audio_length)
    plt.plot(time, testSamples, label="Test Samples")
    plt.plot(time, predictedSamples, label="Predicted Samples")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio timeline")

    plt.show()
    pass

if __name__ == "__main__":
    image_shape = (240,240,3)
    PROCESSED_DATA_FOLDER = "processedData/"    #folder where all pre-processed images are located
    timeSteps = 10

    [
    X_train,
    Y_train,
    X_test,
    Y_test
    ] = loadDataset(PROCESSED_DATA_FOLDER,image_shape,timeSteps=timeSteps,lstm=True)                   #Load the dataset

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)