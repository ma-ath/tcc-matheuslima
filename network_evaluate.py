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

"""
This is a server-only script! It process some data to be visualized in the visualize_prediction script, a client only one.
"""

PROCESSED_DATA_FOLDER = "processedData/"    #folder where all pre-processed images are located
image_shape = (240,240,3)           #input layer receives an RGB 240x240 image
#PLOT_SIZE = 2000
timeSteps = 10

                                    #Check if the model is already in cache
if os.path.isfile(os.path.join('cache', 'architecture.json')) & os.path.isfile(os.path.join('cache', 'model_weights.h5')):
    print("[INFO] Loading cached model ...")
    model = load_model()
else:
    print("[Error] There is no cached model to load. You should run network_train before trying to evaluate")
    raise EnvironmentError

model.compile(optimizer='adam', loss=mean_squared_error)

model.summary()                     #Show network model

[
    X_test,
    Y_test
] = loadDataset_testOnly(PROCESSED_DATA_FOLDER,image_shape,timeSteps=timeSteps,lstm=True)                   #Load the dataset

#score = model.evaluate(X_test, Y_test, verbose=0)
#print('[INFO] Model\'s score: ', score)

#seed()

"""     #Not suitable for LSTM output
print("[INFO] Showing real and predicted value for some random data")
for i in range(10):
    image_index = randrange(0,X_test.shape[0],1)

    X_predict = np.expand_dims(X_test[image_index],0)
  
    prediction = model.predict(X_predict)

    print("Image:" + str(image_index))
    print("Prediction: " + str(prediction))
    print("Real value: " + str(Y_test[image_index]))

print("[INFO] Predicting data for a full array")
"""

Y_predicted = []

# Prepare a predictionSamples vector, in order to plot it
for i in range(X_test.shape[0]):
    X_predict = np.expand_dims(X_test[i],0)
  
    prediction = model.predict(X_predict)

    newshape = (timeSteps,1)

    prediction = prediction[0]
    print(prediction.shape)

    Y_predicted.append(prediction)

Y_predicted = np.array(Y_predicted).astype("float32")

PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]

newshape = (PLOT_SIZE,1)

Y_predicted = np.reshape(Y_predicted,newshape)

Y_test = np.reshape(Y_test,newshape)

print(Y_test)
print(Y_test.shape)

if not os.path.isdir('visualization'):
        os.mkdir('visualization')
'''
Save those data to be vizualized in the client script "visualize_prediction" 
'''
np.save("visualization/visualization-real-lbl.npy",Y_test[0:PLOT_SIZE])
np.save("visualization/visualization-prediction-lbl.npy",Y_predicted)
