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
    from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D
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

    X_train_vgg_features = np.load(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTRAIN)
    X_test_vgg_features = np.load(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTEST)
    #Load the dataset


    telegramSendMessage('Starting the features extraction process for the images with GAP')

    # ----------------- GAP LAYER ----------------- #
    inputs = Input(VGG16_OUTPUT_SHAPE)

    outputs = GlobalAveragePooling2D(data_format=None)(inputs)

    model = Model(inputs=inputs, outputs=outputs)
    # ----------------- GAP LAYER ----------------- #

    model.compile(optimizer='adam', loss='mse') #This doesnt matter, as we are not training the network

    model.summary()

    telegramSendMessage('Extracting features from training dataset[GAP] ...')

    # Extract features for each image (train)
    X_features = []
    number_of_images = X_train_vgg_features.shape[0]
    for index in range(number_of_images):
       
        image = np.expand_dims(X_train_vgg_features[index],0)
    
        feature = model.predict(image)

        X_features.append(feature)
    X_features = np.array(X_features).astype("float32")
    X_features = np.reshape(X_features,(number_of_images,512))
    np.save(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTRAIN+'_GAP',X_features)

    telegramSendMessage('Extracting features from testing dataset[GAP] ...')
    # Extract features for each image (test)
    X_features = []
    number_of_images = X_test_vgg_features.shape[0]
    for index in range(number_of_images):
       
        image = np.expand_dims(X_test_vgg_features[index],0)
    
        feature = model.predict(image)

        X_features.append(feature)
    X_features = np.array(X_features).astype("float32")
    X_features = np.reshape(X_features,(number_of_images,512))
    np.save(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTEST+'_GAP',X_features)
    
    telegramSendMessage('Feature extracting process ended sucefully[GAP]')
    telegramSendMessage('Starting the features extraction process for the images with GMP')

    # ----------------- GMP LAYER ----------------- #
    inputs = Input(VGG16_OUTPUT_SHAPE)

    outputs = GlobalMaxPooling2D(data_format=None)(inputs)

    model = Model(inputs=inputs, outputs=outputs)
    # ----------------- GMP LAYER ----------------- #

    model.compile(optimizer='adam', loss='mse') #This doesnt matter, as we are not training the network

    model.summary()

    telegramSendMessage('Extracting features from training dataset[GMP] ...')

    # Extract features for each image (train)
    X_features = []
    number_of_images = X_train_vgg_features.shape[0]
    for index in range(number_of_images):
       
        image = np.expand_dims(X_train_vgg_features[index],0)
    
        feature = model.predict(image)

        X_features.append(feature)
    X_features = np.array(X_features).astype("float32")
    X_features = np.reshape(X_features,(number_of_images,512))
    np.save(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTRAIN+'_GMP',X_features)

    telegramSendMessage('Extracting features from testing dataset[GMP] ...')
    # Extract features for each image (test)
    X_features = []
    number_of_images = X_test_vgg_features.shape[0]
    for index in range(number_of_images):
       
        image = np.expand_dims(X_test_vgg_features[index],0)
    
        feature = model.predict(image)

        X_features.append(feature)
    X_features = np.array(X_features).astype("float32")
    X_features = np.reshape(X_features,(number_of_images,512))
    np.save(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTEST+'_GMP',X_features)
    
    telegramSendMessage('ALL feature extracting process ended sucefully')

except Exception as e:
    telegramSendMessage('an error has occurred')
    telegramSendMessage(str(e))