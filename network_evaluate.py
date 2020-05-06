import os                                   #
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU

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
from include.auxiliaryFunctions import *

PROCESSED_DATA_FOLDER = "processedData/"    #folder where all pre-processed images are located
BATCH_SIZE = 1
NB_EPOCH = 1

image_shape = (240,240,3)           #input layer receives an RGB 240x240 image
lr_list = [0.001, 0.0003, 9e-05]    #loss rate for the training process (Adam optimizer)

                                    #Check if the model is already in cache
print("Using Cached Model")
model = load_model()
                                    #Model optimizer and compilation
opt = Adam(learning_rate=0.001, epsilon=9e-05, amsgrad=False)
model.compile(optimizer=opt, loss=mean_squared_error)#, metrics=['accuracy'])

model.summary()                     #Show network model

[
    X_train,
    Y_train,
    X_test,
    Y_test
] = loadDataset(PROCESSED_DATA_FOLDER,image_shape)                   #Load the dataset

score = model.evaluate(X_test, Y_test, verbose=2)
print('Model\'s score: ', score)

seed()

for i in range(10):
    image_index = randrange(0,X_test.shape[0],1)

    X_predict = np.expand_dims(X_test[image_index],0)
  
    prediction = model.predict(X_predict)

    print("Image:" + str(image_index) + '\n')
    print("Prediction: " + str(prediction))
    print("Real value: " + str(Y_test[image_index]))



#img = Image.fromarray(X_train[10], 'RGB')
#img.show()