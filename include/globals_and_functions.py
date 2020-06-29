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

# ------------------- GLOBAL_CONSTANTS ------------------- #
#Dataset creation constants
# -------------------------------------------------------- #
# this constant indicates how much of the data will be used as train data
TEST_DATA_RATIO = 0.75
AUDIO_DATA_NAME = "audioPower.npy"
PROCESSED_DATA_FOLDER = "processedData/"
dataset_datapath = "./dataset/" #Datapath for the dataset
dataset_train_datapath = "./dataset/train/"  #Path for train dataset
dataset_test_datapath = "./dataset/test/"    #Path for test dataset   
dataset_raw = "./dataset/raw/"  #Path in which all raw video files are
dataset_config_filename = "config"
dataset_config_train_filename = "config-train"
dataset_config_test_filename = "config-test"
output_resolution = "224x224"   #This string determines the output resolution of the resized video
frameJump = 10                  #This number determines what is the next frame in the amostration process
dataset_test_raw = "./dataset/raw/test/"    #Path in which all raw video files are
dataset_train_raw = "./dataset/raw/train/"  #Path in which all raw video files are
number_of_frames_filename = "number_of_frames"
video_sizes_filename_train = "video_sizes_train"
video_sizes_filename_test = "video_sizes_test"

PROCESSED_DATA_FOLDER = "processedData/"
image_shape = (224,224,3)
timeSteps = 100
timeStepArray = [3,9,27]

# -------------------------------------------------------- #
# ------------------ AUXILIARY FUNCTIONS ----------------- #

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

    # mean2 = np.mean(image_array,axis=(0,1,2))
    # std2 = np.std(image_array,axis=(0,1,2))

    # print("Mean before preprocessin: "+str(mean))
    # print("Standart deviation before preprocessing: "+str(std))
    # print("Mean after preprocessin: "+str(mean2))
    # print("Standart deviation after preprocessing: "+str(std2))

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

def loadDataset(PROCESSED_DATA_FOLDER,image_shape, timeSteps=100,lstm=False,move_window_by_one=False,causal_prediction=True):
    """
    PROCESSED_DATA_FOLDER: simply the nama of the folder where all processed data is (in fact, it is a global constant)
    image_shape: the shape of the image (also a global constant)
    timeSteps: in case of using the LSTM, how many time steps per batch
    lstm: it using LSTM or not (old method)
    move_window_by_one: if using LSTM, should the window move image-per-image or window-by-window (will the windows overlap?)
    causal_prediction: if using LSTM, will it predict the sound volume at the end of the window or in the middle of the window?
    """

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

    X_train = preprocess_image(X_train,usecache=True,train_or_test='train')
    X_test = preprocess_image(X_test,usecache=True,train_or_test='test')

    # This is a temporary solution, i just delete one of the audio sources
    # The ideal thing to do would be to take the mean between the two sources
    Y_train = np.delete(Y_train, -1, axis=1)
    Y_test = np.delete(Y_test, -1, axis=1)

    # If not using LSTM, return is shape (None,224,224,3)
    if lstm == False:
        return X_train,Y_train,X_test,Y_test

    if move_window_by_one == False:
        """
            A big part of this checking is not needed if we process the dataset taking into consideration that the number
            of images should be a multiple of timesteps (which i did after)
        """
        number_images = X_train.shape[0]    #total number of images on dataset
        samples = floor(number_images / timeSteps)  #number of samples for the given number of timeSteps
        throw_away_images = number_images - samples*timeSteps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

        X_new_shape = (samples,timeSteps,)+image_shape    #new input format
        Y_new_shape = (samples,timeSteps,1)               #new output format

        X_train = np.delete(X_train,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor
        Y_train = np.delete(Y_train,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor

        X_train = np.reshape(X_train,X_new_shape)   #reshape the tensor to include the time dimension
        Y_train = np.reshape(Y_train,Y_new_shape)   #reshape the tensor to include the time dimension

        #  ---------------------------  Do same thing to test dataset   -----------------------------#
        number_images = X_test.shape[0]    #total number of images on dataset
        samples = floor(number_images / timeSteps)  #number of samples for the given number of timeSteps
        throw_away_images = number_images - samples*timeSteps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

        X_new_shape = (samples,timeSteps,)+image_shape    #new input format
        Y_new_shape = (samples,timeSteps,1)               #new output format

        X_test = np.delete(X_test,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor
        Y_test = np.delete(Y_test,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor

        X_test = np.reshape(X_test,X_new_shape)   #reshape the tensor to include the time dimension
        Y_test = np.reshape(Y_test,Y_new_shape)   #reshape the tensor to include the time dimension

        # if move_window_by_one == False, return is of shape (None/timeSteps,timeSteps,240,240,3)
    else:
        # This part of the code was base on the Tensorflow LSTM example:
        # https://www.tensorflow.org/tutorials/structured_data/time_series
        #
        
        # Loads the video_sizes array. This array contains what is the size of each video
        # on X_test. Knowing this, we can manage not to mix the videos on transition
        with open(PROCESSED_DATA_FOLDER+video_sizes_filename_train,"rb") as fp:
            number_of_frames = pickle.load(fp)

        number_of_frames = number_of_frames[::-1]   # I have to reverse this array
        if causal_prediction == True:
            target_size = 0     # If causal, we want to predict the audio volume at the last image of the batch
        else:
            target_size = -(timeSteps-1)/2  # If non causal, we want to predict the volume at the center of the batch

        # Window loop
        frame_sum = 0   # This variable keeps track of what frame in X_train is being processed now
        for i in range(len(number_of_frames)):  # For each video in X_train . . .
            print('video'+str(i))
            for j in range(frame_sum,frame_sum+number_of_frames[i]):  # For each frame in this video . . .
                print('frame'+str(j))

                start_index = j+timeSteps
                end_index = j+number_of_frames[i] - target_size
                print('start'+str(start_index))
                print('end'+str(end_index))
                for k in range(start_index, end_index):
                    indices = range(k-timeSteps, k)
                    print(indices)





            frame_sum += number_of_frames[i]

#        # Reshape data from (history_size,) to (history_size, 1)
#        data.append(np.reshape(dataset[indices], (history_size, 1)))
#        labels.append(dataset[i+target_size])

#    return X_train,Y_train,X_test,Y_test

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

def save_model(model,folder_path):
    json_string = model.to_json()
    if not os.path.isdir('cache/'+folder_path):
        os.mkdir('cache/'+folder_path)
    open(os.path.join('cache/'+folder_path, 'architecture.json'), 'w').write(json_string)

def save_weights(model,folder_path):
    if not os.path.isdir('cache/'+folder_path):
        os.mkdir('cache/'+folder_path)
    model.save_weights(os.path.join('cache/'+folder_path, 'model_weights.h5'), overwrite=True)

def load_model(folder_path):
    model = model_from_json(open(os.path.join('cache/'+folder_path, 'architecture.json')).read())
    model.load_weights(os.path.join('cache/'+folder_path, 'model_weights.h5'))
    return model

#Function that returns if a program is installed at the machine
def is_tool(name):
    """
        Check whether `name` is on PATH and marked as executable.
    """

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

def plotAudioPowerWithPrediction(testSamples,predictedSamples,to_file=False,image_path='.'):
    plt.close('all')
    
    plt.figure("Audio Power")

    audio_length = testSamples.shape[0]
    time = np.linspace(0., 1/audio_length ,audio_length)
    plt.plot(time, testSamples, label="Test Samples")
    plt.plot(time, predictedSamples, label="Predicted Samples")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio timeline")

    if to_file == False:
        plt.show()
    else:
        plt.savefig(image_path+'/AudioPowerWithPrediction.png')

    pass

def plotTrainingLoss(history,to_file=False,image_path='.'):
    plt.close('all')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if to_file == False:
        plt.show()
    else:
        plt.savefig(image_path+'/lossPlot.png')
    pass

if __name__ == "__main__":

    [
    X_train,
    Y_train,
    X_test,
    Y_test
    ] = loadDataset(PROCESSED_DATA_FOLDER,image_shape,timeSteps=3,lstm=True,move_window_by_one=True,causal_prediction=True)                   #Load the dataset

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)