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
from include.global_constants import *
from networks import *
from keras.utils.vis_utils import plot_model

"""
    This script run throught all networks variations and trains then one after another    
"""
# -------------------------- DATASET LOAD -------------------------- #
telegramSendMessage('Loading dataset')

#[
#    X_train,
#    Y_train,
#    X_test,
#    Y_test
#] = loadDataset(PROCESSED_DATA_FOLDER,image_shape,timeSteps=timeSteps,lstm=True)                   #Load the dataset

X_train = np.load("postProcessedData/X_train.npy")
Y_train = np.load("postProcessedData/Y_train.npy")
X_test = np.load("postProcessedData/X_test.npy")
Y_test = np.load("postProcessedData/Y_test.npy")

# print("Processando labels!!!")
# Y_train, mean1, std1 = preprocess_labels(Y_train)
# Y_test, mean2, std2 = preprocess_labels(Y_test)

telegramSendMessage('Dataset loaded, network training process started')

# -------------------------- DATASET LOAD -------------------------- #
# ---------------------------- TRAINING ---------------------------- #


for network in networks:

    # Load the desired network model
    model = networkModel(network)

    # Create a folder in cache to save the results from the running test
    # Folder name is specifyed in the model_name key of dictionary
    try:
        if not os.path.exists('./cache/'+network['model_name']):
            os.makedirs('./cache/'+network['model_name'])
    except OSError:
	    print ('Error: Creating directory')
	    exit ()

    # This is the learning rate scheduler, it changes the learning rate of fit
    # depending in the current epoch
    def scheduler(epoch):
        if epoch < 30:
            return network['learning_schedule'][0]
        elif epoch < 50:
            return network['learning_schedule'][1]
        else:
            return network['learning_schedule'][2]

    learning_schedule = LearningRateScheduler(scheduler)

    model_checkpoint = ModelCheckpoint('./cache/'+network['model_name']+'/model_checkpoint.hdf5',
                                            monitor='val_loss',
                                            verbose=2,save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto')

    callback = [learning_schedule,model_checkpoint]

    #Training Optimizer
    opt = network['optimizer']

    #Loss function to minimize
    if network['loss_function'] == 'mse':
        loss_function = mean_squared_error
    else:
        print("[Warning]: loss function does not suported. Using default (mse)")
        loss_function = mean_squared_error

    #Model Compile
    model.compile(optimizer=opt, loss=loss_function) #, metrics=['accuracy'])  #We can not use accuracy as a metric in this model

    #Show network model in terminal and save it to disk
    netconfig_file = open('./cache/'+network['model_name']+'/network_configuration.txt', 'w')
    print('Fitting the following model:')
    netconfig_file.write('Fitting the following model:\n')
    for key, value in network.items():
        print(key, ' : ', value)
        netconfig_file.write(str(key)+' : '+str(value)+'\n')
    model.summary()
    plot_model(model,
                to_file='./cache/'+network['model_name']+'/model_plot.png',
                show_shapes=True,
                show_layer_names=True)
    netconfig_file.close()


    #Fit model

    fit_history = model.fit(
        X_train,
        Y_train,
        batch_size=network['batch_size'],
        epochs=network['number_of_epochs'],
        verbose=2,
        validation_data=(X_test, Y_test),
        callbacks=callback)

    save_model(model,network['model_name'])                   #Save the calculated model to disk
    save_weights(model,network['model_name'])                 #Save the calculated weigths to disk

    #Save the fitting history to disk
    fit_history = pandas.DataFrame(fit_history.history)

    with open('cache/'+network['model_name']+'/fit_history.csv', mode='w') as f:
        fit_history.to_csv(f)
    
    telegramSendMessage('Network '+network['model_name']+' training process ended successfully')

# ---------------------------- TRAINING ---------------------------- #

telegramSendMessage('All network models were trained successfully')