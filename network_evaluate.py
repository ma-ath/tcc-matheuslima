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


def loadDataset():
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
    X_train = np.array(X_train).astype("uint8")
    Y_train = np.array(Y_train).astype("uint8")
    X_test = np.array(X_test).astype("uint8")
    Y_test = np.array(Y_test).astype("uint8")

    return X_train,Y_train,X_test,Y_test

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
model.compile(optimizer=opt, loss=mean_squared_error, metrics=['accuracy'])

model.summary()                     #Show network model

[
    X_train,
    Y_train,
    X_test,
    Y_test
] = loadDataset()                   #Load the dataset

score = model.evaluate(X_test, Y_test, verbose=2)
print('Model\'s score: ', score)

#img = Image.fromarray(X_train[10], 'RGB')
#img.show()