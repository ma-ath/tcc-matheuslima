import os                                   #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import model_from_json
from keras_preprocessing.image import ImageDataGenerator
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
    Y_train = np.array(Y_train).astype("float32")
    X_test = np.array(X_test).astype("uint8")
    Y_test = np.array(Y_test).astype("float32")

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
    #model.load_weights(os.path.join('cache', 'model_weights.h5'))
    return model

PROCESSED_DATA_FOLDER = "processedData/"    #folder where all pre-processed images are located
BATCH_SIZE = 1
NB_EPOCH = 1
USING_CACHE = False

image_shape = (240,240,3)           #input layer receives an RGB 240x240 image
lr_list = [0.001, 0.0003, 9e-05]    #loss rate for the training process (Adam optimizer)

                                    #Check if the model is already in cache
if os.path.isfile(os.path.join('cache', 'architecture.json')) & USING_CACHE == True:
    print("using cached model")
    model = load_model()
else:
    model = networkModel(image_shape)   #model created by Leonardo Mazza, modified by me
    save_model(model)
                                        #Adam optimizer
opt = Adam(learning_rate=0.001, epsilon=9e-05, amsgrad=False)

model.compile(optimizer=opt, loss=mean_squared_error) #, metrics=['accuracy'])  #We can not use accuracy as a metric in this model
model.summary()                     #Show network model

[
    X_train,
    Y_train,
    X_test,
    Y_test
] = loadDataset()                   #Load the dataset

                                    #Fit model
fit_history = model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=2,validation_data=(X_test, Y_test))

# construct the training image generator for data augmentation
# This is made so as to help generalize the dataset
# dataset_augumentation = ImageDataGenerator(rotation_range=0,zoom_range=0,
# 	width_shift_range=0, height_shift_range=0, shear_range=0,
# 	horizontal_flip=False,fill_mode="nearest")
# train the network
# //I use the fit_generator method in order to not overload the RAM memory in the server, as the
# //model.fit statement has to load the full dataset in RAM.
# This actually makes no sense here, because I already loaded the dataset into ram before
# fit_history = model.fit_generator(dataset_augumentation.flow(X_train, Y_train, batch_size=BATCH_SIZE),
#	validation_data=(X_test, Y_test),steps_per_epoch=len(X_train),
#   epochs=NB_EPOCH)

save_model(model)                   #Save the calculated model to disk
save_weights(model)                 #Save the calculated weigths to disk

                                    #Save the fitting history to disk
fit_history = pandas.DataFrame(fit_history.history)

with open('fit_history.csv', mode='w') as f:
    fit_history.to_csv(f)