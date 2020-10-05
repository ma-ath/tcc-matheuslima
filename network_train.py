from include.telegram_logger import *
from include.globals_and_functions import *
from network_model import networkModel
from networks import *
from folds import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_GPU

os.environ["PATH"] += os.pathsep + "/usr/bin/dot"



from tensorflow import keras
from keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import numpy as np
from PIL import Image
import re
from matplotlib import pyplot as plt
import pandas
import gc

from keras.utils.vis_utils import plot_model

def loadDataset(Fold_name,
                CNN = "vgg16",
                Pooling = "GAP",
                LSTM = True,
                time_steps = 3,
                overlap_windows = True,
                causal_prediction = True):

    """
    Function that loads the dataset to the training process, using any struture

    Fold_name:          From which fold should the function load the dataset
    CNN, Pooling:       Which features should the function use to create the dataset
    LSTM:               If the dataset loaded is being used in a lstm network
    time_steps:         How many frame inputs are there in one window of the LSTM
    overlap_windows:    If the window move "one-by-one", or "time_steps-by-time_steps"
    causal_prediction:  If the predicted audio sample is in the middle of the window (non-causal), or at the end of the window (causal)
    """

    #   Try loading the processed dataset from file:
    if (CNN == "vgg16"):
        dataset_datapath = CONST_STR_DATASET_FOLDS_DATAPATH+CONST_STR_DATASET_FEATURES_VGG16
    elif (CNN == "inceptionV3"):
        dataset_datapath = CONST_STR_DATASET_FOLDS_DATAPATH+CONST_STR_DATASET_FEATURES_INCEPTIONV3
    elif (CNN == "resnet50"):
        dataset_datapath = CONST_STR_DATASET_FOLDS_DATAPATH+CONST_STR_DATASET_FEATURES_RESNET50
    elif (CNN == None):
        dataset_datapath = CONST_STR_DATASET_FOLDS_DATAPATH
    else:
        print_error("CNN not suported for dataset loading")
        telegramSendMessage("[ERRO] CNN not suported for dataset loading")
        exit(1)

    if not (CNN == None):
        training_images_filename = dataset_datapath+"input_training_data_"+Pooling+"_"+Fold_name+".npy"
        testing_images_filename = dataset_datapath+"input_testing_data_"+Pooling+"_"+Fold_name+".npy"
        training_labels_filename = CONST_STR_DATASET_FOLDS_DATAPATH+"output_training_data_"+Fold_name+".npy"
        testing_labels_filename = CONST_STR_DATASET_FOLDS_DATAPATH+"output_testing_data_"+Fold_name+".npy"
    else:
        training_images_filename = dataset_datapath+"input_training_data_"+Fold_name+".npy"
        testing_images_filename = dataset_datapath+"input_testing_data_"+Fold_name+".npy"
        training_labels_filename = CONST_STR_DATASET_FOLDS_DATAPATH+"output_training_data_"+Fold_name+".npy"
        testing_labels_filename = CONST_STR_DATASET_FOLDS_DATAPATH+"output_testing_data_"+Fold_name+".npy"

    number_of_frames_train_filename = CONST_STR_DATASET_FOLDS_DATAPATH+"nof_train_"+Fold_name+".npy"
    number_of_frames_test_filename = CONST_STR_DATASET_FOLDS_DATAPATH+"nof_test_"+Fold_name+".npy"

    try:
        training_images = np.load(training_images_filename)
        testing_images = np.load(testing_images_filename)
        training_labels = np.load(training_labels_filename)
        testing_labels = np.load(testing_labels_filename)
    except:
        print_error("Could not open one or more of the following files:")
        print("\t"+training_images_filename)
        print("\t"+testing_images_filename)
        print("\t"+training_labels_filename)
        print("\t"+testing_labels_filename)
        telegramSendMessage("[ERRO]: Could not find correct dataset on loadDataset")
        exit(1)
    try:
        training_nof = np.load(number_of_frames_train_filename)
        testing_nof = np.load(number_of_frames_test_filename)
    except:
        print_error("Could not open one or more of the following files:")
        print("\t"+number_of_frames_train_filename)
        print("\t"+number_of_frames_test_filename)
        telegramSendMessage("[ERRO]: Could not find nof file for dataset on loadDataset")
        exit(1)

    #   If not using a LSTM network, loading dataset from file is all we need to do
    if not LSTM:
        print_info("Dataset loaded sucessefully")
        return training_images, training_labels, testing_images, testing_labels

    if overlap_windows == False:
        #   We rarelly will use overlap_windows = False, but here is the code
        #
        #   When not overlaying windows, all we need to do is reshape da dataset. Do do that, the number of
        #   images has to be a multiple of time_steps
        #   To do that by, simply deleting some images
        # ------  ------ #
        samples = floor(training_images.shape[0] / time_steps)  #number of samples for the given number of time_steps
        throw_away_images = training_images.shape[0] - samples*time_steps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

        training_images = np.delete(training_images, slice(0, throw_away_images), axis=0)  #trunc the 4d tensor
        training_labels = np.delete(training_labels, slice(0, throw_away_images), axis=0)  #trunc the 4d tensor

        samples = floor(testing_images.shape[0] / time_steps)  #number of samples for the given number of time_steps
        throw_away_images = testing_images.shape[0] - samples*time_steps   #number of images i'll have to throw away (trunc) in order to reshape the 4d tensor in the required 5d tensor

        testing_images = np.delete(testing_images, slice(0, throw_away_images), axis=0)  #trunc the 4d tensor
        testing_labels = np.delete(testing_labels, slice(0, throw_away_images), axis=0)  #trunc the 4d tensor
        # ----------------------------------------------------------------------- #

        samples_train = int(training_images.shape[0] / time_steps)
        samples_test = int(testing_images.shape[0] / time_steps)

        if CNN == None:
            #   We rarelly will use CNN == None, but here is the code
            training_images = np.reshape(training_images, (samples_train, time_steps)+CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE)
            testing_images = np.reshape(testing_images, (samples_test, time_steps)+CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE)

        elif Pooling == "None":
            if CNN == "vgg16":
                training_images = np.reshape(training_images, (samples_train, time_steps)+CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE)
                testing_images = np.reshape(testing_images, (samples_test, time_steps)+CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE)
            elif CNN == "resnet50":
                training_images = np.reshape(training_images, (samples_train, time_steps)+CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE)
                testing_images = np.reshape(testing_images, (samples_test, time_steps)+CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE)
            elif CNN == "inceptionV3":
                training_images = np.reshape(training_images, (samples_train, time_steps)+CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE)
                testing_images = np.reshape(testing_images, (samples_test, time_steps)+CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE)
        else:
            if CNN == "vgg16":
                training_images = np.reshape(training_images, (samples_train, time_steps, CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[2]))
                testing_images = np.reshape(testing_images, (samples_test, time_steps, CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[2]))
            elif CNN == "resnet50":
                training_images = np.reshape(training_images, (samples_train, time_steps, CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[2]))
                testing_images = np.reshape(testing_images, (samples_test, time_steps, CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[2]))
            elif CNN == "inceptionV3":
                training_images = np.reshape(training_images, (samples_train, time_steps, CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[2]))
                testing_images = np.reshape(testing_images, (samples_test, time_steps, CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[2]))

        training_labels = np.reshape(training_labels, (samples_train, time_steps))
        testing_labels = np.reshape(testing_labels, (samples_test, time_steps))

        return training_images, training_labels, testing_images, testing_labels
    
    else:
        # This part of the code was base on the Tensorflow LSTM example:
        # https://www.tensorflow.org/tutorials/structured_data/time_series
        #
        if causal_prediction == True:
            target_size = 0     # If causal, we want to predict the audio volume at the last image of the batch
        else:
            target_size = int((time_steps-1)/2)  # If non causal, we want to predict the volume at the center of the batch


        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        # ----------------------- TRAINS SET ----------------------- # 

        # Window loop
        frame_sum = 0   # This variable keeps track of what frame in training_images is being processed now
        for i in range(len(training_nof)):  # For each video in training_images . . .

            start_index = frame_sum+time_steps
            end_index = frame_sum+training_nof[i]
            for j in range(start_index, end_index):     # For each window in this video . . .
                indices = range(j-time_steps, j)
                
                if CNN == None:
                    X_train.append(np.reshape(training_images[indices], (time_steps,)+CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE))
                elif Pooling == None:
                    if CNN == "vgg16":
                        X_train.append(np.reshape(training_images[indices], (time_steps,)+CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE))
                    elif CNN == "resnet50":
                        X_train.append(np.reshape(training_images[indices], (time_steps,)+CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE))
                    elif CNN == "inceptionV3":
                        X_train.append(np.reshape(training_images[indices], (time_steps,)+CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE))
                elif (Pooling == 'GAP') or (Pooling == 'GMP'):
                    if CNN == "vgg16":
                        X_train.append(np.reshape(training_images[indices], (time_steps, CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[2])))
                    elif CNN == "resnet50":
                        X_train.append(np.reshape(training_images[indices], (time_steps, CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[2])))
                    elif CNN == "inceptionV3":
                        X_train.append(np.reshape(training_images[indices], (time_steps, CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[2])))
                Y_train.append(training_labels[j-target_size])

            frame_sum += training_nof[i]
        # -----------------------TEST SET ----------------------- # 
        # Window loop
        frame_sum = 0   # This variable keeps track of what frame in testing_images is being processed now
        for i in range(len(testing_nof)):  # For each video in testing_images . . .

            start_index = frame_sum+time_steps
            end_index = frame_sum+testing_nof[i]
            for j in range(start_index, end_index):     # For each window in this video . . .
                indices = range(j-time_steps, j)
                
                if CNN == None:
                    X_test.append(np.reshape(testing_images[indices], (time_steps,)+CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE))
                elif Pooling == None:
                    if CNN == "vgg16":
                        X_test.append(np.reshape(testing_images[indices], (time_steps,)+CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE))
                    elif CNN == "resnet50":
                        X_test.append(np.reshape(testing_images[indices], (time_steps,)+CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE))
                    elif CNN == "inceptionV3":
                        X_test.append(np.reshape(testing_images[indices], (time_steps,)+CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE))
                elif (Pooling == 'GAP') or (Pooling == 'GMP'):
                    if CNN == "vgg16":
                        X_test.append(np.reshape(testing_images[indices], (time_steps, CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[2])))
                    elif CNN == "resnet50":
                        X_test.append(np.reshape(testing_images[indices], (time_steps, CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[2])))
                    elif CNN == "inceptionV3":
                        X_test.append(np.reshape(testing_images[indices], (time_steps, CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[2])))
                Y_test.append(testing_labels[j-target_size])

            frame_sum += testing_nof[i]


        X_train = np.array(X_train).astype("float32")
        Y_train = np.array(Y_train).astype("float32")
        X_test = np.array(X_test).astype("float32")
        Y_test = np.array(Y_test).astype("float32")

        #   Do a manual memory free in these arrays

        del testing_images
        del training_images
        del testing_labels
        del training_labels
        gc.collect()

        return X_train, Y_train, X_test, Y_test
try:
    """
        This script run throught all networks and folds variations and trains then one after another
    """

    # ---------------------------- TRAINING ---------------------------- #
    for fold in folds:
        print_info("Stating training in fold "+str(fold['number']))
        telegramSendMessage("Stating training in fold "+str(fold['number']))

        for network in networks:

            results_datapath = CONST_STR_RESULTS_DATAPATH+fold['name']+'/'+network['model_name']

            if not os.path.isfile(results_datapath+'/lossPlot.png'):    #If this file exists, this test was already done
                # -------------------------- DATASET LOAD -------------------------- #
                print_info("Loading "+fold['name']+" dataset for model "+network['model_name'])
                telegramSendMessage("Loading "+fold['name']+" dataset for model "+network['model_name'])

                [
                X_train,
                Y_train,
                X_test,
                Y_test
                ] = loadDataset(Fold_name=fold['name'],
                                CNN=network['cnn'],
                                Pooling=network['pooling'],
                                LSTM=network['lstm'],
                                time_steps=network['time_steps'],
                                overlap_windows=network['overlap_windows'],
                                causal_prediction=network['causal_prediction'])

                telegramSendMessage('Starting training process for '+network['model_name'])
            
                # -------------------------- DATASET LOAD -------------------------- #

                # Load the desired network model
                model = networkModel(network)

                # Create a folder in cache to save the results from the running test
                # Folder name is specifyed in the model_name key of dictionary

                try:
                    if not os.path.exists(results_datapath):
                        os.makedirs(results_datapath)
                except OSError:
                    print_error('Error: Creating directory to save training data')
                    exit(1)

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

                model_checkpoint = ModelCheckpoint(results_datapath+'/model_checkpoint.hdf5',
                                                    monitor='val_loss',
                                                    verbose=2,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='auto')

                callback = [learning_schedule, model_checkpoint]

                #Training Optimizer
                opt = network['optimizer']

                #Loss function to minimize
                if network['loss_function'] == 'mse':
                    loss_function = mean_squared_error
                else:
                    print_warning("Loss function specified in model is not suported. Using default (mse)")
                    loss_function = mean_squared_error

                #Model Compile
                model.compile(optimizer=opt, loss=loss_function) #  We can not use accuracy as a metric in this model

                #Show network model in terminal and save it to disk
                netconfig_file = open(results_datapath+'/network_configuration.txt', 'w')
                print_info('Fitting the following model:')
                netconfig_file.write('Fitting the following model:\n')
                for key, value in network.items():
                    print('\t'+key,': ', value)
                    netconfig_file.write('\t'+str(key)+': '+str(value)+'\n')
                model.summary()
                #  Plot model has given me too many "pydot` failed to call GraphViz" errors. It's not so important
                #
                #  plot_model(model,
                #            to_file=results_datapath+'/model_plot.png',
                #            show_shapes=True,
                #            show_layer_names=True)
                netconfig_file.close()

                #Fit model

                fit_history = model.fit(
                    X_train,
                    Y_train,
                    batch_size=network['batch_size'],
                    epochs=network['epochs'],
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    callbacks=callback)

                telegramSendMessage('Network '+network['model_name']+' training process ended successfully')

                save_model(model, results_datapath)                   #Save the calculated model to disk
                save_weights(model, results_datapath)                 #Save the calculated weigths to disk

                #Save the fitting history to disk
                fit_history_df = pandas.DataFrame(fit_history.history)

                with open(results_datapath+'/fit_history.csv', mode='w') as f:
                    fit_history_df.to_csv(f)
                
                telegramSendMessage('Saving vizualization data for '+network['model_name'])
                # # ------------ SAVE SOME VISUALIZATION DATA ------------ #

                # # ------------------- predicte over test set ------------------- #
                print_info("Predicting output for test data over last epoch")
                #Y_predicted = []
                #Y_vtest = Y_test
                #
                # Prepare a predictionSamples vector, in order to plot it
                #for i in range(X_test.shape[0]):
                #    X_predict = np.expand_dims(X_test[i], 0)
                #    prediction = model.predict(X_predict, batch_size=network['batch_size'])
                #    newshape = (network['time_steps'], 1)
                #    prediction = prediction[0]
                #    Y_predicted.append(prediction)
                #Y_predicted = np.array(Y_predicted).astype("float32")
                Y_predicted = model.predict(X_test, batch_size=network['batch_size'])
                PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]
                newshape = (PLOT_SIZE, 1)
                Y_predicted = np.reshape(Y_predicted, newshape)
                Y_vtest = np.reshape(Y_test, newshape)

                np.save(results_datapath+'/res_real_lastepoch_test.npy', Y_vtest[0:PLOT_SIZE])
                np.save(results_datapath+'/res_prediction_lastepoch_test.npy', Y_predicted)

                plotAudioPowerWithPrediction(Y_vtest, Y_predicted, to_file=True, image_path=results_datapath,image_name='/prediction_Test_lastepoch.png')
                # ------------------- predicte over train set ------------------- #
                """
                print_info("Predicting output for train data over last epoch")
                Y_predicted = []
                Y_vtest = Y_train

                # Prepare a predictionSamples vector, in order to plot it
                for i in range(X_train.shape[0]):
                    X_predict = np.expand_dims(X_train[i], 0)
                    prediction = model.predict(X_predict, batch_size=network['batch_size'])
                    newshape = (network['time_steps'], 1)
                    prediction = prediction[0]
                    Y_predicted.append(prediction)
                Y_predicted = np.array(Y_predicted).astype("float32")
                PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]
                newshape = (PLOT_SIZE, 1)
                Y_predicted = np.reshape(Y_predicted, newshape)

                Y_vtest = np.reshape(Y_train, newshape)

                np.save(results_datapath+'/res_real_lastepoch_train.npy', Y_vtest[0:PLOT_SIZE])
                np.save(results_datapath+'/res_prediction_lastepoch_train.npy', Y_predicted)

                plotAudioPowerWithPrediction(Y_vtest, Y_predicted, to_file=True, image_path=results_datapath, image_name='/prediction_Train_lastepoch.png')
                """
                # # ------------------- Load best checkpoint weigths ------------------- # #
                model.load_weights(os.path.join(results_datapath, 'model_checkpoint.hdf5'))
                # # ------------------- Load best checkpoint weigths ------------------- # #
                # # ------------------- predicte over test set ------------------- #
                print_info("Predicting output for test data over best checkpoint")
                #Y_predicted = []
                #Y_vtest = Y_test
                #
                # Prepare a predictionSamples vector, in order to plot it
                #for i in range(X_test.shape[0]):
                #    X_predict = np.expand_dims(X_test[i], 0)
                #    prediction = model.predict(X_predict, batch_size=network['batch_size'])
                #    newshape = (network['time_steps'], 1)
                #    prediction = prediction[0]
                #    Y_predicted.append(prediction)
                #Y_predicted = np.array(Y_predicted).astype("float32")
                Y_predicted = model.predict(X_test, batch_size=network['batch_size'])
                PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]
                newshape = (PLOT_SIZE, 1)
                Y_predicted = np.reshape(Y_predicted, newshape)
                Y_vtest = np.reshape(Y_test, newshape)

                np.save(results_datapath+'/res_real_checkpoint_test.npy', Y_vtest[0:PLOT_SIZE])
                np.save(results_datapath+'/res_prediction_checkpoint_test.npy', Y_predicted)

                plotAudioPowerWithPrediction(Y_vtest, Y_predicted, to_file=True, image_path=results_datapath, image_name='/prediction_Test_checkpoint.png')
                # ------------------- predicte over train set ------------------- #
                """
                print_info("Predicting output for train data over best checkpoint")
                Y_predicted = []
                Y_vtest = Y_train

                # Prepare a predictionSamples vector, in order to plot it
                for i in range(X_train.shape[0]):
                    X_predict = np.expand_dims(X_train[i], 0)
                    prediction = model.predict(X_predict, batch_size=network['batch_size'])
                    newshape = (network['time_steps'], 1)
                    prediction = prediction[0]
                    Y_predicted.append(prediction)
                Y_predicted = np.array(Y_predicted).astype("float32")
                PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]
                newshape = (PLOT_SIZE, 1)
                Y_predicted = np.reshape(Y_predicted, newshape)

                Y_vtest = np.reshape(Y_train, newshape)

                np.save(results_datapath+'/res_real_checkpoint_train.npy', Y_vtest[0:PLOT_SIZE])
                np.save(results_datapath+'/res_prediction_checkpoint_train.npy', Y_predicted)

                plotAudioPowerWithPrediction(Y_vtest, Y_predicted, to_file=True, image_path=results_datapath, image_name='/prediction_Train_checkpoint.png')
                """
                # summarize history for loss
                plotTrainingLoss(fit_history, to_file=True, image_path=results_datapath)
                #------------ SAVE SOME VISUALIZATION DATA ------------ #
                #------------ FREE MEMORY ------------ #
                del X_train
                del X_test
                del Y_train
                del Y_test
                del Y_vtest
                del Y_predicted
                gc.collect()
                #------------ FREE MEMORY ------------ #
    # ---------------------------- TRAINING ---------------------------- #
    print_info('All network models were trained successfully')
    telegramSendMessage('All network models were trained successfully')
    exit(0)
except Exception as e:

    print_error('An error has occurred')
    print_error(str(e))
    telegramSendMessage('[ERROR]: An error has occurred')
    telegramSendMessage(str(e))