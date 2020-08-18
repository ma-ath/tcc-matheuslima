try:
    import os                                   #
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    from tensorflow import keras
    from keras.optimizers import Adam
    from keras.losses import mean_squared_error
    from include.globals_and_functions import *
    from include.telegram_logger import *
    import numpy as np
    from networks import *

    #telegramSendMessage('loading dataset')

    #[
    #    X_train,
    #    Y_train,
    #    X_test,
    #    Y_test
    #] = loadDatasetFromCache()  #Load the dataset

    #telegramSendMessage('dataset load')

    MODELOS = [ 'model_math_lstm_0',
                'model_math_lstm_1',
                'model_math_lstm_2',
                'model_math_lstm_3',
                'model_math_lstm_4',
                'model_math_lstm_5',
                'model_math_lstm_6',
                'model_math_lstm_7',
                'model_math_lstm_8',
                'model_math_lstm_9',
                'model_math_lstm_10',
                'model_math_lstm_11',
                'model_math_lstm_12',
                'model_math_lstm_13',
                'model_math_lstm_14',
                'model_math_lstm_15',
                'model_math_lstm_16',
                'model_math_lstm_17',
                'model_math_lstm_18',
                'model_math_lstm_19',
                'model_math_lstm_20',
                'model_math_lstm_21',
                'model_math_lstm_22',
                'model_math_lstm_23',
                'model_math_lstm_24',
                'model_math_lstm_25',
                'model_math_lstm_26',
                'model_math_lstm_27',
                'model_math_lstm_28',
                'model_math_lstm_29',
                'model_math_lstm_30',
                'model_math_lstm_31'
        ]
    
    iteracao = 0

    for MODELO in MODELOS:

        telegramSendMessage('Loading dataset for model '+networks[iteracao]['model_name'])

        [
        X_train, 
        Y_train,
        X_test,
        Y_test
        ] = loadDatasetLSTM(causal_prediction=networks[iteracao]['dataset_causal_prediction'],
                        overlap_windows=networks[iteracao]['dataset_overlap_windows'],
                        timeSteps=networks[iteracao]['time_steps'],
                        features_only=networks[iteracao]['features_input'],
                        pooling_input=networks[iteracao]['pooling_input'])

        telegramSendMessage('Starting training process for '+networks[iteracao]['model_name'])

        iteracao+=1


        telegramSendMessage('carregando modelo '+MODELO)
        path = os.path.join('cache/'+MODELO, 'architecture.json')
        telegramSendMessage('load model: '+path)
        model = model_from_json(open(os.path.join('cache/'+MODELO+'/', 'architecture.json')).read())
        model.load_weights(os.path.join('cache/'+MODELO, 'model_checkpoint.hdf5'))

        model.compile(optimizer='adam', loss=mean_squared_error)

        model.summary()

        Y_predicted = []
        Y_vtest = Y_test

        # Prepare a predictionSamples vector, in order to plot it
        for i in range(X_test.shape[0]):
            X_predict = np.expand_dims(X_test[i],0)

            prediction = model.predict(X_predict)

            newshape = (9,1)

            prediction = prediction[0]

            Y_predicted.append(prediction)

        Y_predicted = np.array(Y_predicted).astype("float32")

        PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]

        newshape = (PLOT_SIZE,1)

        Y_predicted = np.reshape(Y_predicted,newshape)

        Y_vtest = np.reshape(Y_test,newshape)

        plotAudioPowerWithPrediction(Y_vtest,Y_predicted,to_file=True,image_path='cache/'+MODELO,image_name='/prediction_val_checkpoint.png')

        # ------------------- predicte over train set ------------------- #
        Y_predicted = []
        Y_vtest = Y_train

        # Prepare a predictionSamples vector, in order to plot it
        for i in range(X_train.shape[0]):
            X_predict = np.expand_dims(X_train[i],0)

            prediction = model.predict(X_predict)

            newshape = (9,1)

            prediction = prediction[0]

            Y_predicted.append(prediction)

        Y_predicted = np.array(Y_predicted).astype("float32")

        PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]

        newshape = (PLOT_SIZE,1)

        Y_predicted = np.reshape(Y_predicted,newshape)

        Y_vtest = np.reshape(Y_train,newshape)

        plotAudioPowerWithPrediction(Y_vtest,Y_predicted,to_file=True,image_path='cache/'+MODELO,image_name='/prediction_train_checkpoint.png')

    telegramSendMessage('script end')
except Exception as e:

    telegramSendMessage('an error has occurred')
    telegramSendMessage(str(e))