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
NB_EPOCH = 2
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

model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5',
                                            monitor='val_loss',
                                            verbose=2,save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto')

callback = [learning_schedule,model_checkpoint]

                                        #Adam optimizer
opt = Adam(learning_rate=0.001, epsilon=9e-05, amsgrad=False)

model.compile(optimizer=opt, loss=mean_squared_error) #, metrics=['accuracy'])  #We can not use accuracy as a metric in this model
model.summary()                     #Show network model

[
    X_train,
    Y_train,
    X_test,
    Y_test
] = loadDataset(PROCESSED_DATA_FOLDER,image_shape)                   #Load the dataset

telegramSendMessage('Network training process started')
                                    #Fit model
fit_history = model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=2,validation_data=(X_test, Y_test),callbacks=callback)

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

telegramSendMessage('Network training process ended successfully')