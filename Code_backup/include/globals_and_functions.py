import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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

DATASET_CACHE_FOLDER = "./cache/cached_dataset/"
DATASET_VGG16_IMAGEFEATURES_FILEPATH = 'vgg16_image_features/'
DATASET_VGG16_IMAGEFEATURES_FTRAIN = 'X_ftrain.npy'
DATASET_VGG16_IMAGEFEATURES_FTEST = 'X_ftest.npy'
VGG16_OUTPUT_SHAPE = (7,7,512)

image_shape = (224,224,3)
#timeSteps = 3
timeStepArray = [3,9,27]

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# -------------------------------------------------------- #
# ------------------ AUXILIARY FUNCTIONS ----------------- #

def preprocess_image(image_array, usecache=False,train_or_test='train'):
    # We only need to calculate those values if we are dealing with a new dataset. So to speedup things, we can use the precalculated mean and std for train and test datasets

    #image_array[:, :, :, 0] /= 255
    #image_array[:, :, :, 1] /= 255
    #image_array[:, :, :, 2] /= 255
    """
        Bellow means and stds were calculated with the 'calculate_std_mean_byparts' script.
        Dataset used was the 38x2 (38 videos on train, 2 on test)
    """
    if not usecache:
        mean = np.mean(image_array,axis=(0,1,2))
        std = np.std(image_array,axis=(0,1,2))
    elif train_or_test == 'train':
        mean = [90.13959802, 86.7031525, 94.3868386]
        std = [58.67927085, 52.2398956, 54.57580831]
    elif train_or_test == 'test':
        mean = [73.57426842, 74.66081132, 88.61860926]
        std = [59.68105447, 49.92547875, 52.9416586 ]
    else:
        raise ValueError

    image_array[:, :, :, 0] -= (mean[0])
    image_array[:, :, :, 1] -= (mean[1])
    image_array[:, :, :, 2] -= (mean[2])

    image_array[:, :, :, 0] /= (std[0])
    image_array[:, :, :, 1] /= (std[1])
    image_array[:, :, :, 2] /= (std[2])

    return image_array

def loadDataset(test_only=False):

    """
    This is a simple function that loads the dataset
    """
    if test_only == False:
        X_train = np.load(PROCESSED_DATA_FOLDER+"images_training-img.npy",mmap_mode="r")
        Y_train = np.load(PROCESSED_DATA_FOLDER+"images_training-lbl.npy",mmap_mode="r")
    X_test = np.load(PROCESSED_DATA_FOLDER+"images_testing-img.npy",mmap_mode="r")
    Y_test = np.load(PROCESSED_DATA_FOLDER+"images_testing-lbl.npy",mmap_mode="r")

    if test_only == False:
        X_train = np.reshape(X_train,(X_train.shape[0],)+image_shape).astype("float32")
    X_test = np.reshape(X_test,(X_test.shape[0],)+image_shape).astype("float32")

    #Normalize the input image to have "0" mean and standart deviation of "1"
    # I tried to do it using the builtin keras function "preprocess_input()", but doing it by hand has better results
    # I do it before loading the dataset because if done before, i would have to save the
    # training file as float32, which takes considerably more space than a uint8 format

    if test_only == False:
        X_train = preprocess_image(X_train,usecache=False,train_or_test='train')
    X_test = preprocess_image(X_test,usecache=False,train_or_test='test')

    # This is a temporary solution, i just delete one of the audio sources
    # The ideal thing to do would be to take the mean between the two sources
    if test_only == False:
        Y_train = np.delete(Y_train, -1, axis=1)
    Y_test = np.delete(Y_test, -1, axis=1)

    # For some reason, axis 3 (colour) is fliped
    X_test = np.flip( X_test, axis=3 )
    if test_only == False:
        X_train = np.flip( X_train, axis=3 )
        return X_train,Y_train,X_test,Y_test

    return X_test,Y_test

def loadDatasetLSTM(timeSteps=3,overlap_windows=False,causal_prediction=True,features_only=False,pooling_input=None):
    """
    Function that loads the dataset to the training process, for a lstm structure

    timeSteps : how many frames inputs are there in one window for the LSTM
    overlay_windows : if the window move "one-by-one", or "many-by-many"
    causal_prediction: If the predicted audio sample is in the middle of the window (non-causal), or at the end of the window(causal)
    features_only : if instead of the actual image input (frames,224,224,3), we use directily the VGG16 extracted feature maps (frames,7,7,512)
    pooling_input : if using features_only, if the input has a pre-processed pooling input (frames,512) or not (frames,7,7,512) 
    """
    if features_only == False:
        training_images = np.load(PROCESSED_DATA_FOLDER+"images_training-img.npy")
        testing_images = np.load(PROCESSED_DATA_FOLDER+"images_testing-img.npy")
    elif pooling_input == None:
        training_images = np.load(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTRAIN)
        testing_images = np.load(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTEST)
    elif pooling_input == "GAP":
        training_images = np.load(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTRAIN+'_GAP.npy')
        testing_images = np.load(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTEST+'_GAP.npy')
    elif pooling_input == "GMP":
        training_images = np.load(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTRAIN+'_GMP.npy')
        testing_images = np.load(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTEST+'_GMP.npy')

    training_labels = np.load(PROCESSED_DATA_FOLDER+"images_training-lbl.npy")
    testing_labels = np.load(PROCESSED_DATA_FOLDER+"images_testing-lbl.npy")


    # First reshape is made only for preprocesseing the image array
    if features_only == False:
        training_images = np.reshape(training_images,(training_images.shape[0],)+image_shape).astype("float32")
        testing_images = np.reshape(testing_images,(testing_images.shape[0],)+image_shape).astype("float32")

        training_images = preprocess_image(training_images)
        testing_images = preprocess_image(testing_images)

    # I delete one of the audio sources
    training_labels = np.delete(training_labels, -1, axis=1)
    testing_labels = np.delete(testing_labels, -1, axis=1)

    if overlap_windows == False:
        #
        # A big part of this checking is not needed if we process the dataset taking into consideration that the number
        # of images should be a multiple of timesteps (which i did after)
        #
        # ------ ADAPT THE DATASET TO RESHAPE IN CASE USING OTHER TIMESTEP ------ #
        samples = floor(training_images.shape[0] / timeSteps)  #number of samples for the given number of timeSteps
        throw_away_images = training_images.shape[0] - samples*timeSteps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

        training_images = np.delete(training_images,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor
        training_labels = np.delete(training_labels,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor

        samples = floor(testing_images.shape[0] / timeSteps)  #number of samples for the given number of timeSteps
        throw_away_images = testing_images.shape[0] - samples*timeSteps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

        testing_images = np.delete(testing_images,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor
        testing_labels = np.delete(testing_labels,slice(0,throw_away_images),axis=0)  #trunc the 4d tensor
        # ----------------------------------------------------------------------- #

        samples_train = int(training_images.shape[0] / timeSteps) # samples will ALWAYS be a multiple of timeSteps (3,9 or 27), because i force it when processing the raw files
        samples_test = int(testing_images.shape[0] / timeSteps)

        if features_only == False:
            X_train = np.reshape(training_images,(samples_train,timeSteps)+image_shape)
            X_test = np.reshape(testing_images,(samples_test,timeSteps)+image_shape)
        elif pooling_input == None:
            X_train = np.reshape(training_images,(samples_train,timeSteps)+VGG16_OUTPUT_SHAPE)
            X_test = np.reshape(testing_images,(samples_test,timeSteps)+VGG16_OUTPUT_SHAPE)
        elif pooling_input == 'GAP' or 'GMP':
            X_train = np.reshape(training_images,(samples_train,timeSteps,VGG16_OUTPUT_SHAPE[2]))
            X_test = np.reshape(testing_images,(samples_test,timeSteps,VGG16_OUTPUT_SHAPE[2]))

        Y_train = np.reshape(training_labels,(samples_train,timeSteps))
        Y_test = np.reshape(testing_labels,(samples_test,timeSteps))

        if pooling_input == None:
            X_train = np.flip( X_train, axis=4 )
            X_test = np.flip( X_test, axis=4 )

        # if move_window_by_one == False, return is of shape (None/timeSteps,timeSteps,224,224,3)
    else:
        # This part of the code was base on the Tensorflow LSTM example:
        # https://www.tensorflow.org/tutorials/structured_data/time_series
        #
        if features_only == False:
            training_images = np.reshape(training_images,(training_images.shape[0],image_shape[0]*image_shape[1]*image_shape[2]))
            testing_images = np.reshape(testing_images,(testing_images.shape[0],image_shape[0]*image_shape[1]*image_shape[2]))
        elif pooling_input == None:
            training_images = np.reshape(training_images,(training_images.shape[0],VGG16_OUTPUT_SHAPE[0]*VGG16_OUTPUT_SHAPE[1]*VGG16_OUTPUT_SHAPE[2]))
            testing_images = np.reshape(testing_images,(testing_images.shape[0],VGG16_OUTPUT_SHAPE[0]*VGG16_OUTPUT_SHAPE[1]*VGG16_OUTPUT_SHAPE[2]))
        elif pooling_input == 'GAP' or 'GMP':
            training_images = np.reshape(training_images,(training_images.shape[0],VGG16_OUTPUT_SHAPE[2]))
            testing_images = np.reshape(testing_images,(testing_images.shape[0],VGG16_OUTPUT_SHAPE[2]))


        if causal_prediction == True:
            target_size = 0     # If causal, we want to predict the audio volume at the last image of the batch
        else:
            target_size = int((timeSteps-1)/2)  # If non causal, we want to predict the volume at the center of the batch

        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        # ----------------------- TRAINS SET ----------------------- # 
        # Loads the video_sizes array. This array contains what is the size of each video
        # on X_test. Knowing this, we can manage not to mix the videos on transition
        with open(PROCESSED_DATA_FOLDER+video_sizes_filename_train,"rb") as fp:
            number_of_frames = pickle.load(fp)

        number_of_frames = number_of_frames[::-1]   # I have to reverse this array

        # Window loop
        frame_sum = 0   # This variable keeps track of what frame in X_train is being processed now
        for i in range(len(number_of_frames)):  # For each video in X_train . . .

            start_index = frame_sum+timeSteps
            end_index = frame_sum+number_of_frames[i]
            for j in range(start_index, end_index):     # For each window in this video . . .
                indices = range(j-timeSteps, j)
                
                if features_only == False:
                    X_train.append(np.reshape(training_images[indices],(timeSteps,)+image_shape))
                elif pooling_input == None:
                    X_train.append(np.reshape(training_images[indices],(timeSteps,)+VGG16_OUTPUT_SHAPE))
                elif pooling_input == 'GAP' or 'GMP':
                    X_train.append(np.reshape(training_images[indices],(timeSteps,VGG16_OUTPUT_SHAPE[2])))
                Y_train.append(training_labels[j-target_size])

            frame_sum += number_of_frames[i]
        # -----------------------TEST SET ----------------------- # 
        # Loads the video_sizes array. This array contains what is the size of each video
        # on X_test. Knowing this, we can manage not to mix the videos on transition
        with open(PROCESSED_DATA_FOLDER+video_sizes_filename_test,"rb") as fp:
            number_of_frames = pickle.load(fp)

        number_of_frames = number_of_frames[::-1]   # I have to reverse this array

        # Window loop

        frame_sum = 0   # This variable keeps track of what frame in X_train is being processed now
        for i in range(len(number_of_frames)):  # For each video in X_train . . .

            start_index = frame_sum+timeSteps
            end_index = frame_sum+number_of_frames[i]
            for j in range(start_index, end_index):     # For each window in this video . . .
                indices = range(j-timeSteps, j)

                if features_only == False:
                    X_test.append(np.reshape(testing_images[indices],(timeSteps,)+image_shape))
                elif pooling_input == None:
                    X_test.append(np.reshape(testing_images[indices],(timeSteps,)+VGG16_OUTPUT_SHAPE))
                elif pooling_input == 'GAP' or 'GMP':
                    X_test.append(np.reshape(testing_images[indices],(timeSteps,VGG16_OUTPUT_SHAPE[2])))
                Y_test.append(testing_labels[j-target_size])

            frame_sum += number_of_frames[i]

        X_train = np.array(X_train).astype("float32")
        Y_train = np.array(Y_train).astype("float32")
        X_test = np.array(X_test).astype("float32")
        Y_test = np.array(Y_test).astype("float32")

        # For some reason channels red and blue (axis 4) are flipped
        if pooling_input == None:
            X_train = np.flip( X_train, axis=4 )
            X_test = np.flip( X_test, axis=4 )

    return X_train,Y_train,X_test,Y_test

def loadDatasetFromCache():
    try:
        X_train = np.load(DATASET_CACHE_FOLDER+"X_train.npy")
        Y_train = np.load(DATASET_CACHE_FOLDER+"Y_train.npy")
        X_test = np.load(DATASET_CACHE_FOLDER+"X_test.npy")
        Y_test = np.load(DATASET_CACHE_FOLDER+"Y_test.npy")

        return X_train,Y_train,X_test,Y_test

    except:
        print('Error: There is no cached dataset')
        exit()

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

def plotAudioPowerWithPrediction(testSamples,predictedSamples,to_file=False,image_path='.',image_name='/AudioPowerWithPrediction.png'):
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
        plt.savefig(image_path+image_name)

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
    ] = loadDatasetLSTM(causal_prediction=False,overlap_windows=True,timeSteps=9,features_only=True,pooling_input='GAP')   #Load the dataset

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)