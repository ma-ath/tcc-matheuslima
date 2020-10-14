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
CONST_STR_DATASET_RAW_DATAPATH = "./dataset/raw/"   #Path in which all raw video files are
CONST_STR_DATASET_CONFIG_FILENAME = "config"        #this is the name of the config file for raw videos
CONST_STR_DATASET_DATAPATH = "./dataset/"           #Datapath for the dataset
CONST_STR_DATASET_OUTPUT_RESOLUTION = "224x224"     #Output resolution of video for
CONST_VEC_DATASET_OUTPUT_RESOLUTION = (224, 224)
CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE = (224, 224,3)
CONST_INT_DATASET_DECIMATION_FACTOR = 10            #This number determines factor of decimation when extracting frames from video
CONST_STR_DATASET_NMB_OF_FRAMES_FILENAME = "nmb_of_frames"
CONS_STR_DATASET_AUDIOFILE_FILENAME = "/audioData.wav"
CONS_STR_DATASET_AUDIODATA_FILENAME = "/audioData.npy"
CONS_STR_DATASET_STATISTICS_FILENAME = "/statistics"
CONS_STR_DATASET_STACKED_FRAMES_FILENAME = "/imagedata.npy"
CONST_STR_DATASET_FOLDS_DATAPATH = "./dataset/folds/"
CONST_STR_DATASET_FEATURES_VGG16 = "vgg16/"
CONST_STR_DATASET_FEATURES_RESNET50 = "resnet50/"
CONST_STR_DATASET_FEATURES_INCEPTIONV3 = "inceptionV3/"

CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE = (7, 7, 512)
CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE = (5, 5, 2048)
CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE = (7, 7, 2048)

CONST_STR_RESULTS_DATAPATH = "./results/"

CUDA_GPU = "0"

# -------------------------------------------------------- #
# ------------------ AUXILIARY FUNCTIONS ----------------- #
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def print_info(s):
    print(f"{bcolors.OKGREEN}[INFO]: {bcolors.ENDC}"+s)
def print_error(s):
    print(f"{bcolors.FAIL}[ERROR]: {bcolors.ENDC}"+s)
def print_warning(s):
    print(f"{bcolors.WARNING}[WARNING]: {bcolors.ENDC}"+s)

def save_model(model, folder_path):
    json_string = model.to_json()
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    open(os.path.join(folder_path, 'architecture.json'), 'w').write(json_string)

def save_weights(model, folder_path, filename=None):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    if filename is None:
        model.save_weights(os.path.join(folder_path, 'model_weights.h5'), overwrite=True)
    else:
        model.save_weights(os.path.join(folder_path, filename), overwrite=True)
        
def load_model(folder_path):
    model = model_from_json(open(os.path.join(folder_path, 'architecture.json')).read())
    model.load_weights(os.path.join(folder_path, 'model_weights.h5'))
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

def plotTrainingLoss(history, to_file=False, image_path='.'):
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
