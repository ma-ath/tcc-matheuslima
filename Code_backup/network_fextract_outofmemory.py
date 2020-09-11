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
        The dataset is normalized with mean and std calculated seperatelly    
    """

    try:
        if not os.path.exists(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH):
            os.makedirs(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH)
    except OSError:
        print ('Error: Creating directory')
        raise

    # ----------------- VGG16 FREEZE ----------------- #
    inputs = Input(image_shape)

    convolutional_layer = VGG16(weights='imagenet', include_top=False,input_shape=image_shape)
    for layer in convolutional_layer.layers[:]:
        layer.trainable = False     #Freezes all layers in the vgg16
    
    outputs = convolutional_layer(inputs)

    #Model used to extract all features
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse') #This doesnt matter, as we are not training the network
    model.summary()
    # ----------------- VGG16 FREEZE ----------------- #

    # -------------------------- FEATURE EXTRACTION -------------------------- #
    telegramSendMessage('Initializing network_extract_features script')

    # Load dataset in reading nmap_mode (doesn't load to memory) 
    X_train = np.load(PROCESSED_DATA_FOLDER+"images_training-img.npy",mmap_mode="r")
    Y_train = np.load(PROCESSED_DATA_FOLDER+"images_training-lbl.npy",mmap_mode="r")
    X_test = np.load(PROCESSED_DATA_FOLDER+"images_testing-img.npy",mmap_mode="r")
    Y_test = np.load(PROCESSED_DATA_FOLDER+"images_testing-lbl.npy",mmap_mode="r")

    Y_train = np.delete(Y_train, -1, axis=1)
    Y_test = np.delete(Y_test, -1, axis=1)

    number_of_images = X_train.shape[0]

    # Extract features for each image (train)
    X_features = []
    number_of_images = X_train.shape[0]

    telegramSendMessage('Extracting features from '+str(number_of_images)+' images')
    telegramSendMessage('Extracting features from training dataset')

    for index in range(number_of_images):
        X_train_slice = X_train[index]

        X_train_slice = np.expand_dims(X_train_slice,0)
        X_train_slice = np.reshape(X_train_slice,(X_train_slice.shape[0],)+image_shape).astype("float32")
        X_train_slice = preprocess_image(X_train_slice,usecache=True,train_or_test='train')
        X_train_slice = np.flip(X_train_slice, axis=3)
        
        feature = model.predict(X_train_slice)

        X_features.append(feature)
    X_features = np.array(X_features).astype("float32")
    X_features = np.reshape(X_features,(number_of_images,)+VGG16_OUTPUT_SHAPE)
    np.save(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTRAIN,X_features)

    telegramSendMessage('Extracting features from testing dataset ...')
    # Extract features for each image (test)
    X_features = []
    number_of_images = X_test.shape[0]
    for index in range(number_of_images):
        X_train_slice = X_test[index]

        X_train_slice = np.expand_dims(X_train_slice,0)
        X_train_slice = np.reshape(X_train_slice,(X_train_slice.shape[0],)+image_shape).astype("float32")
        X_train_slice = preprocess_image(X_train_slice,usecache=True,train_or_test='test')
        X_train_slice = np.flip(X_train_slice, axis=3)

        feature = model.predict(X_train_slice)

        X_features.append(feature)
    X_features = np.array(X_features).astype("float32")
    X_features = np.reshape(X_features,(number_of_images,)+VGG16_OUTPUT_SHAPE)
    np.save(PROCESSED_DATA_FOLDER+DATASET_VGG16_IMAGEFEATURES_FILEPATH+DATASET_VGG16_IMAGEFEATURES_FTEST,X_features)
    
    telegramSendMessage('Feature extracting process ended sucefully')

except Exception as e:
    telegramSendMessage('an error has occurred')
    telegramSendMessage(str(e))