from include.telegram_logger import *
from include.globals_and_functions import *
from network_model import networkModel
from networks import *
from folds import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_GPU

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

from keras.utils.vis_utils import plot_model

#b,c,d,e = loadDataset("fold_1", CNN = None, Pooling = "GAP", LSTM = True, overlap_windows=False)

#print(b.shape)
#print(c.shape)
#print(d.shape)
#print(e.shape)

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
    """
        ### NADA AQUI ESTÁ FEITO

        # This part of the code was base on the Tensorflow LSTM example:
        # https://www.tensorflow.org/tutorials/structured_data/time_series
        #
        if features_only == False:
            training_images = np.reshape(training_images,(training_images.shape[0],image_shape[0]*image_shape[1]*image_shape[2]))
            testing_images = np.reshape(testing_images,(testing_images.shape[0],image_shape[0]*image_shape[1]*image_shape[2]))
        elif pooling_input == None:
            training_images = np.reshape(training_images,(training_images.shape[0],VGG16_OUTPUT_SHAPE[0]*VGG16_OUTPUT_SHAPE[1]*VGG16_OUTPUT_SHAPE[2]))
            testing_images = np.reshape(testing_images,(testing_images.shape[0],VGG16_OUTPUT_SHAPE[0]*VGG16_OUTPUT_SHAPE[1]*VGG16_OUTPUT_SHAPE[2]))
        elif pooling_input == 'GAP' or 'GMP':
            training_images = np.reshape(training_images,(training_images.shape[0],VGG16_OUTPUT_SHAPE[2]))
            testing_images = np.reshape(testing_images,(testing_images.shape[0],VGG16_OUTPUT_SHAPE[2]))


        if causal_prediction == True:
            target_size = 0     # If causal, we want to predict the audio volume at the last image of the batch
        else:
            target_size = int((timeSteps-1)/2)  # If non causal, we want to predict the volume at the center of the batch

        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        # ----------------------- TRAINS SET ----------------------- # 
        # Loads the video_sizes array. This array contains what is the size of each video
        # on X_test. Knowing this, we can manage not to mix the videos on transition
        with open(PROCESSED_DATA_FOLDER+video_sizes_filename_train,"rb") as fp:
            number_of_frames = pickle.load(fp)

        number_of_frames = number_of_frames[::-1]   # I have to reverse this array

        # Window loop
        frame_sum = 0   # This variable keeps track of what frame in X_train is being processed now
        for i in range(len(number_of_frames)):  # For each video in X_train . . .

            start_index = frame_sum+timeSteps
            end_index = frame_sum+number_of_frames[i]
            for j in range(start_index, end_index):     # For each window in this video . . .
                indices = range(j-timeSteps, j)
                
                if features_only == False:
                    X_train.append(np.reshape(training_images[indices],(timeSteps,)+image_shape))
                elif pooling_input == None:
                    X_train.append(np.reshape(training_images[indices],(timeSteps,)+VGG16_OUTPUT_SHAPE))
                elif pooling_input == 'GAP' or 'GMP':
                    X_train.append(np.reshape(training_images[indices],(timeSteps,VGG16_OUTPUT_SHAPE[2])))
                Y_train.append(training_labels[j-target_size])

            frame_sum += number_of_frames[i]
        # -----------------------TEST SET ----------------------- # 
        # Loads the video_sizes array. This array contains what is the size of each video
        # on X_test. Knowing this, we can manage not to mix the videos on transition
        with open(PROCESSED_DATA_FOLDER+video_sizes_filename_test,"rb") as fp:
            number_of_frames = pickle.load(fp)

        number_of_frames = number_of_frames[::-1]   # I have to reverse this array

        # Window loop

        frame_sum = 0   # This variable keeps track of what frame in X_train is being processed now
        for i in range(len(number_of_frames)):  # For each video in X_train . . .

            start_index = frame_sum+timeSteps
            end_index = frame_sum+number_of_frames[i]
            for j in range(start_index, end_index):     # For each window in this video . . .
                indices = range(j-timeSteps, j)

                if features_only == False:
                    X_test.append(np.reshape(testing_images[indices],(timeSteps,)+image_shape))
                elif pooling_input == None:
                    X_test.append(np.reshape(testing_images[indices],(timeSteps,)+VGG16_OUTPUT_SHAPE))
                elif pooling_input == 'GAP' or 'GMP':
                    X_test.append(np.reshape(testing_images[indices],(timeSteps,VGG16_OUTPUT_SHAPE[2])))
                Y_test.append(testing_labels[j-target_size])

            frame_sum += number_of_frames[i]

        X_train = np.array(X_train).astype("float32")
        Y_train = np.array(Y_train).astype("float32")
        X_test = np.array(X_test).astype("float32")
        Y_test = np.array(Y_test).astype("float32")

        # For some reason channels red and blue (axis 4) are flipped
        if pooling_input == None:
            X_train = np.flip( X_train, axis=4 )
            X_test = np.flip( X_test, axis=4 )
    """
    #return X_train,Y_train,X_test,Y_test


try:
    print(a)
    """
        This script run throught all networks and folds variations and trains then one after another
    """

    # ---------------------------- TRAINING ---------------------------- #
    for fold in folds:
        print_info("Stating training in fold "+fold['number'])
        telegramSendMessage("Stating training in fold "+fold['number'])

        for network in networks:
            # -------------------------- DATASET LOAD -------------------------- #
            print_info("Loading "+fold['name']+" dataset for model "+network['model_name'])
            telegramSendMessage("Loading "+fold['name']+" dataset for model "+network['model_name'])

            [
            X_train, 
            Y_train,
            X_test,
            Y_test
            ] = loadDatasetLSTM(causal_prediction=network['dataset_causal_prediction'],
                            overlap_windows=network['dataset_overlap_windows'],
                            timeSteps=network['time_steps'],
                            features_only=network['features_input'],
                            pooling_input=network['pooling_input'])

            telegramSendMessage('Starting training process for '+network['model_name'])
        
            # -------------------------- DATASET LOAD -------------------------- #

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

            telegramSendMessage('Network '+network['model_name']+' training process ended successfully')

            save_model(model,network['model_name'])                   #Save the calculated model to disk
            save_weights(model,network['model_name'])                 #Save the calculated weigths to disk

            #Save the fitting history to disk
            fit_history_df = pandas.DataFrame(fit_history.history)

            with open('cache/'+network['model_name']+'/fit_history.csv', mode='w') as f:
                fit_history_df.to_csv(f)
            
            telegramSendMessage('Saving vizualization data for '+network['model_name'])
            # # ------------ SAVE SOME VISUALIZATION DATA ------------ #

            # # ------------------- predicte over test set ------------------- #
            Y_predicted = []
            Y_vtest = Y_test

            # Prepare a predictionSamples vector, in order to plot it
            for i in range(X_test.shape[0]):
                X_predict = np.expand_dims(X_test[i],0)
        
                prediction = model.predict(X_predict,batch_size=network['batch_size'])

                newshape = (network['time_steps'],1)

                prediction = prediction[0]

                Y_predicted.append(prediction)

            Y_predicted = np.array(Y_predicted).astype("float32")

            PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]

            newshape = (PLOT_SIZE,1)

            Y_predicted = np.reshape(Y_predicted,newshape)

            Y_vtest = np.reshape(Y_test,newshape)

            np.save('cache/'+network['model_name']+'/visualization-real-lbl.npy',Y_vtest[0:PLOT_SIZE])
            np.save('cache/'+network['model_name']+'/visualization-prediction-lbl.npy',Y_predicted)

            plotAudioPowerWithPrediction(Y_vtest,Y_predicted,to_file=True,image_path='cache/'+network['model_name'])

            # ------------------- predicte over train set ------------------- #
            Y_predicted = []
            Y_vtest = Y_train

            # Prepare a predictionSamples vector, in order to plot it
            for i in range(X_train.shape[0]):
                X_predict = np.expand_dims(X_train[i],0)
        
                prediction = model.predict(X_predict,batch_size=network['batch_size'])

                newshape = (network['time_steps'],1)

                prediction = prediction[0]

                Y_predicted.append(prediction)

            Y_predicted = np.array(Y_predicted).astype("float32")

            PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]

            newshape = (PLOT_SIZE,1)

            Y_predicted = np.reshape(Y_predicted,newshape)

            Y_vtest = np.reshape(Y_train,newshape)

            np.save('cache/'+network['model_name']+'/visualization-real-train-lbl.npy',Y_vtest[0:PLOT_SIZE])
            np.save('cache/'+network['model_name']+'/visualization-prediction-train-lbl.npy',Y_predicted)

            plotAudioPowerWithPrediction(Y_vtest,Y_predicted,to_file=True,image_path='cache/'+network['model_name'],image_name='/prediction_Train.png')

            plotTrainingLoss(fit_history,to_file=True,image_path='cache/'+network['model_name'])

            # summarize history for loss

            #------------ SAVE SOME VISUALIZATION DATA ------------ #    


    # ---------------------------- TRAINING ---------------------------- #

    telegramSendMessage('All network models were trained successfully')
except Exception as e:

    telegramSendMessage('an error has occurred')
    telegramSendMessage(str(e))