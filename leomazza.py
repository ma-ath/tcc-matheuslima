import os                                   #
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU

from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from network_model import networkModel
import numpy as np
from PIL import Image
import re
from matplotlib import pyplot as plt

def loadImagesDataset():
    imageData = []

    #   Load images in alphabetical order #####
    myimages = [] #list of image filenames
    dirFiles = os.listdir('./dataset/images/') #list of directory files
    dirFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
    sorted(dirFiles) #sort numerically in ascending order

    for files in dirFiles: #filter out all non jpgs
        if '.jpg' in files:
            myimages.append(files)
    for i in myimages:
        im=Image.open("./dataset/images/"+i)
        imageData.append(im)

    inputData = np.zeros((len(imageData),240,240,3))

    for i in range(0,len(imageData)):
        inputData[i] = np.array(imageData[i]) # Transform from Pillow array to a Numpy array

    return inputData


image_shape = (240,240,3)           #input layer receives an RGB 240x240 image
lr_list = [0.001, 0.0003, 9e-05]    #loss rate for the training process (Adam optimizer)

model = networkModel(image_shape)   #model created by Leonardo Mazza

opt = Adam(lr=lr_list)              #Adam optimizer

model.compile(optimizer=opt, loss=mean_squared_error)

model.summary()
