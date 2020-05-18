import os                                   #
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU

from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error, mean_absolute_error
from keras.models import model_from_json
from network_model import networkModel
import numpy as np
from PIL import Image
import re
from matplotlib import pyplot as plt
import pandas
from random import randrange, seed
from include.auxiliaryFunctions import *

PROCESSED_DATA_FOLDER = "processedData/"    #folder where all pre-processed images are located
BATCH_SIZE = 1
NB_EPOCH = 1
USING_CACHE = False
PLOT_SIZE = 92

image_shape = (240,240,3)           #input layer receives an RGB 240x240 image
lr_list = [0.001, 0.0003, 9e-05]    #loss rate for the training process (Adam optimizer)

                                    #Check if the model is already in cache
#Check if the model is already in cache
if os.path.isfile(os.path.join('cache', 'architecture.json')) & USING_CACHE == True:
    print("using cached model")
    model = load_model()
else:
    model = networkModel(image_shape)   #model created by Leonardo Mazza, modified by me
    save_model(model)
                                    #Model optimizer and compilation
#opt = Adam(learning_rate=0.001, epsilon=9e-05, amsgrad=False)
model.compile(optimizer='adam', loss=mean_squared_error)#, metrics=['accuracy'])

model.summary()                     #Show network model

[
    X_test,
    Y_test
] = loadDataset_testOnly(PROCESSED_DATA_FOLDER,image_shape)                   #Load the dataset

Y_predicted = []

# Prepare a predictionSamples vector, in order to plot it
for i in range(PLOT_SIZE):
    X_predict = np.expand_dims(X_test[i],0)
  
    prediction = model.predict(X_predict)

    Y_predicted.append(prediction)

Y_predicted = np.array(Y_predicted).astype("float32")
Y_predicted = np.squeeze(Y_predicted, axis=(2))


# Y_test,a,b = preprocess_labels(Y_test)

plotAudioPowerWithPrediction(Y_test[0:PLOT_SIZE],Y_predicted)