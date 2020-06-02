import os                                   #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import model_from_json
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from network_model import networkModel
import numpy as np
from PIL import Image
import re
from matplotlib import pyplot as plt
import pandas
from include.telegram_logger import *
from include.auxiliaryFunctions import *

PROCESSED_DATA_FOLDER = "processedData/"    #folder where all pre-processed images are located
BATCH_SIZE = 1
NB_EPOCH = 5

image_shape = (240,240,3)           #input layer receives an RGB 240x240 image
timeSteps = 10
lr_list = [0.001, 0.0003, 9e-05]    #loss rate for the training process (Adam optimizer)

model = networkModel(image_shape,timeSteps)   #model created by Leonardo Mazza, coded by me

# This is the learning rate scheduler, it changes the learning rate of fit
# depending in the current epoch
def scheduler(epoch):
    if epoch < 30:
        return lr_list[0]
    elif epoch < 50:
        return lr_list[1]
    else:
        return lr_list[2]

learning_schedule = LearningRateScheduler(scheduler)

model_checkpoint = ModelCheckpoint('./cache/model_checkpoint.hdf5',
                                            monitor='val_loss',
                                            verbose=2,save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto')

callback = [learning_schedule,model_checkpoint]

                                        #Adam optimizer
#opt = Adam(learning_rate=0.001, epsilon=9e-05, amsgrad=False)

model.compile(optimizer='adam', loss=mean_squared_error) #, metrics=['accuracy'])  #We can not use accuracy as a metric in this model
model.summary()                     #Show network model

[
    X_train,
    Y_train,
    X_test,
    Y_test
] = loadDataset(PROCESSED_DATA_FOLDER,image_shape,timeSteps=timeSteps,lstm=True)                   #Load the dataset

# print("Processando labels!!!")
# Y_train, mean1, std1 = preprocess_labels(Y_train)
# Y_test, mean2, std2 = preprocess_labels(Y_test)

telegramSendMessage('Network training process started')
                                    #Fit model
fit_history = model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=2,validation_data=(X_test, Y_test),callbacks=callback)

save_model(model)                   #Save the calculated model to disk
save_weights(model)                 #Save the calculated weigths to disk

                                    #Save the fitting history to disk
fit_history = pandas.DataFrame(fit_history.history)

with open('cache/fit_history.csv', mode='w') as f:
    fit_history.to_csv(f)

telegramSendMessage('Network training process ended successfully')