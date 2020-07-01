try:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import os                                   #
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    from include.globals_and_functions import *
    from networks import *
    from keras.utils.vis_utils import plot_model

    """
        This script run throught all networks variations and trains then one after another    
    """
    # -------------------------- DATASET LOAD -------------------------- #
    telegramSendMessage('Loading dataset')

    [
        X_train,
        Y_train,
        X_test,
        Y_test
    ] = loadDatasetFromCache()  #Load the dataset
    # -------------------------- DATASET LOAD -------------------------- #
    # ---------------------------- TRAINING ---------------------------- #

    for network in networks:

        telegramSendMessage('Starting training process for '+network['model_name'])

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
        # ------------ SAVE SOME VISUALIZATION DATA ------------ #

        # ------------------- predicte over test set ------------------- #
        Y_predicted = []
        Y_vtest = Y_test

        # Prepare a predictionSamples vector, in order to plot it
        for i in range(X_test.shape[0]):
            X_predict = np.expand_dims(X_test[i],0)
    
            prediction = model.predict(X_predict)

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
    
            prediction = model.predict(X_predict)

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


        # ------------ SAVE SOME VISUALIZATION DATA ------------ #    


    # ---------------------------- TRAINING ---------------------------- #

    telegramSendMessage('All network models were trained successfully')
except Exception as e:

    telegramSendMessage('an error has occurred')
    telegramSendMessage(str(e))