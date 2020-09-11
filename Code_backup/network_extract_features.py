try:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import os                                   #
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    from tensorflow import keras
    import numpy as np
    from include.telegram_logger import *
    from include.globals_and_functions import *
    from keras.applications.vgg16 import VGG16
    from keras.layers import Input
    from keras.models import Model

    """
        This script loads all image inputs and extract only its features with the vgg16 imagenet weights;
        This somewhat solves the OOM problem of loading the full dataset into ram for LSTMs    
    """

    try:
        if not os.path.exists(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH):
            os.makedirs(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH)
    except OSError:
        print ('Error: Creating directory')
        raise

    # -------------------------- DATASET LOAD -------------------------- #
    telegramSendMessage('Loading dataset')

    [
        X_train,
        Y_train,
        X_test,
        Y_test
    ] = loadDataset()  #Load the dataset


    telegramSendMessage('Starting the features extraction process for the images')

    # ----------------- VGG16 FREEZE ----------------- #
    inputs = Input(image_shape)

    convolutional_layer = VGG16(weights='imagenet', include_top=False,input_shape=image_shape)
    for layer in convolutional_layer.layers[:]:
        layer.trainable = False     #Freezes all layers in the vgg16
    
    outputs = convolutional_layer(inputs)

    model = Model(inputs=inputs, outputs=outputs)
    # ----------------- VGG16 FREEZE ----------------- #

    model.compile(optimizer='adam', loss='mse') #This doesnt matter, as we are not training the network

    model.summary()

    # Extract features for each image (train)
    X_features = []
    number_of_images = X_train.shape[0]
    for index in range(number_of_images):
       
        image = np.expand_dims(X_train[index],0)
    
        feature = model.predict(image)

        X_features.append(feature)
    X_features = np.array(X_features).astype("float32")
    X_features = np.reshape(X_features,(number_of_images,)+VGG16_OUTPUT_SHAPE)
    np.save(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTRAIN,X_features)

    telegramSendMessage('Extracting features from testing dataset ...')
    # Extract features for each image (test)
    X_features = []
    number_of_images = X_test.shape[0]
    for index in range(number_of_images):
       
        image = np.expand_dims(X_test[index],0)
    
        feature = model.predict(image)

        X_features.append(feature)
    X_features = np.array(X_features).astype("float32")
    X_features = np.reshape(X_features,(number_of_images,)+VGG16_OUTPUT_SHAPE)
    np.save(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTEST,X_features)
    
    telegramSendMessage('Feature extracting process ended sucefully')

except Exception as e:
    telegramSendMessage('an error has occurred')
    telegramSendMessage(str(e))