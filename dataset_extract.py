from include.globals_and_functions import *
from include.telegram_logger import telegramSendMessage
from folds import *

from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications import InceptionV3
from keras.applications import ResNet50
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model

import argparse
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_GPU
import numpy as np
import gc

def extract_folds_features(argv):
    #   Parse command line to read with which network should the script extract image information
    parser = argparse.ArgumentParser(description='Receives the desired network model for extraction')
    parser.add_argument('--network', '-n', default="vgg16", help='The desired network. We only support vgg16, inceptionV3 and resnet50')
    parser.add_argument('--pooling', '-p', default="GAP", help='Used pooling layer. We support: Global Max Poolling [GAP], Global Average Pooling[GMP], None [None]')
    parser.add_argument('--fold', '-f', default="all", help='Select which fold to extract features')
    args = parser.parse_args()
    
    #   Check if network is suported
    #   We now only have vgg16
    if not((args.network == "vgg16") or (args.network == "inceptionV3") or (args.network == "resnet50")):
        print_error("We currently only suport the followuing models: vgg16, inceptionv3 and resnet50")
        exit(1)
    
    if not ((args.pooling == "GAP") or (args.pooling == "GMP") or (args.pooling == "None")):
        print_error("We currently only suport GAP, GMP and None layers")
        exit(1)
    
    #   ------------------  Create extracting model

    inputs = Input(CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE)

        #   ------------------  Convolution  ------------------
    if args.network == "vgg16":
        convolutional_layer = VGG16(weights='imagenet', include_top=False, input_shape=CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE)
        for layer in convolutional_layer.layers[:]:
            layer.trainable = False     #Freezes all layers in the vgg16
        output_shape = CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE
    if args.network == "inceptionV3":
        convolutional_layer = InceptionV3(weights='imagenet', include_top=False, input_shape=CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE)
        for layer in convolutional_layer.layers[:]:
            layer.trainable = False     #Freezes all layers in the vgg16
        output_shape = CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE
    if args.network == "resnet50":
        convolutional_layer = ResNet50(weights='imagenet', include_top=False, input_shape=CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE)
        for layer in convolutional_layer.layers[:]:
            layer.trainable = False     #Freezes all layers in the vgg16
        output_shape = CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE
    
    convolution_output = convolutional_layer(inputs)

        #   ------------------  Pooling  ------------------
    if args.pooling == "None":
        outputs = convolution_output
    elif args.pooling == "GAP":
        outputs = GlobalAveragePooling2D(data_format=None)(convolution_output)
    elif args.pooling == "GMP":
        outputs = GlobalMaxPooling2D(data_format=None)(convolution_output)

    model = Model(inputs=inputs, outputs=outputs)

    print_info("Extracting network summary:")
    model.summary()


    #   ------------------  Extracting process

    telegramSendMessage("Starting extracting process with network "+args.network+" and pooling layer "+args.pooling)
    print_info("Starting extracting process with network "+args.network+" and pooling layer "+args.pooling)

    #   Create folder to host all extracted models
    if (args.network == "vgg16"):
        extraction_datapath = os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_FOLDS_DATAPATH,CONST_STR_DATASET_FEATURES_VGG16)
    if (args.network == "inceptionV3"):
        extraction_datapath = os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_FOLDS_DATAPATH,CONST_STR_DATASET_FEATURES_INCEPTIONV3)
    if (args.network == "resnet50"):
        extraction_datapath = os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_FOLDS_DATAPATH,CONST_STR_DATASET_FEATURES_RESNET50)

    try:
        if not os.path.exists(extraction_datapath):
            os.makedirs(extraction_datapath)
    except OSError:
        print_error("Could not make directory for features extraction")
        telegramSendMessage("Error: Creating directory")
        exit(1)

    if args.fold == "all":
        for fold in folds:
            #   Load fold dataset
            print_info("Loading dataset from "+fold["name"])
            telegramSendMessage("Loading dataset from "+fold["name"])

            try:
                input_train_data = np.load(os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_FOLDS_DATAPATH,"input_training_data_"+fold['name']+".npy"))
                input_test_data = np.load(os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_FOLDS_DATAPATH,"input_testing_data_"+fold['name']+".npy"))
            except:
                print_error("Could not find dataset. Did you build it?")
                telegramSendMessage("Could not find dataset. Did you build it?")
                exit(1)
            
            #   Extracting features of fold
            print_info("Extracting training features")
            telegramSendMessage("Extracting training features")

            train_features = []

            number_of_images = input_train_data.shape[0]
            
            for index in range(number_of_images):
                #   Expand dimention of image from (224,224,3) to (1,224,224,3)
                image = np.expand_dims(input_train_data[index], 0)

                #   Pass image throught the model
                image_feature = model.predict(image)

                #   Append to the train_features array
                train_features.append(image_feature)
            
            #Transform array into ndarray
            train_features = np.array(train_features).astype("float32")
            train_features = np.reshape(train_features, (number_of_images, output_shape[2]))

            #Save the extracted features
            print_info("Saving training features")
            telegramSendMessage("Saving training features")
            np.save(os.path.join(extraction_datapath,"input_training_data_"+args.pooling+"_"+fold["name"]), train_features)

            ###   Repeat to test dataset
            print_info("Extracting testing features")
            telegramSendMessage("Extracting testing features")

            test_features = []
            number_of_images = input_test_data.shape[0]
            for index in range(number_of_images):
                image = np.expand_dims(input_test_data[index], 0)
                image_feature = model.predict(image)
                test_features.append(image_feature)
            test_features = np.array(test_features).astype("float32")
            test_features = np.reshape(test_features, (number_of_images, output_shape[2]))

            #Save the extracted features
            print_info("Saving testing features")
            telegramSendMessage("Saving testing features")
            np.save(os.path.join(extraction_datapath,"input_testing_data_"+args.pooling+"_"+fold["name"]), test_features)

            #   Forcefully delete input datas from memory
            del input_train_data
            del input_test_data
            del train_features
            del test_features
            gc.collect()
    else:
        fold_name = args.fold
        #   Load fold dataset
        print_info("Loading dataset from "+fold_name)
        telegramSendMessage("Loading dataset from "+fold_name)

        try:
            input_train_data = np.load(os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_FOLDS_DATAPATH,"input_training_data_"+fold_name+".npy"))
            input_test_data = np.load(os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_FOLDS_DATAPATH,"input_testing_data_"+fold_name+".npy"))
        except:
            print_error("Could not find dataset. Did you build it?")
            telegramSendMessage("Could not find dataset. Did you build it?")
            exit(1)
        
        #   Extracting features of fold
        print_info("Extracting training features")
        telegramSendMessage("Extracting training features")

        train_features = []

        number_of_images = input_train_data.shape[0]
        
        for index in range(number_of_images):
            #   Expand dimention of image from (224,224,3) to (1,224,224,3)
            image = np.expand_dims(input_train_data[index], 0)

            #   Pass image throught the model
            image_feature = model.predict(image)

            #   Append to the train_features array
            train_features.append(image_feature)
        
        #Transform array into ndarray
        train_features = np.array(train_features).astype("float32")
        train_features = np.reshape(train_features, (number_of_images, output_shape[2]))

        #Save the extracted features
        print_info("Saving training features")
        telegramSendMessage("Saving training features")
        np.save(os.path.join(extraction_datapath,"input_training_data_"+args.pooling+"_"+fold_name), train_features)

        ###   Repeat to test dataset
        print_info("Extracting testing features")
        telegramSendMessage("Extracting testing features")

        test_features = []
        number_of_images = input_test_data.shape[0]
        for index in range(number_of_images):
            image = np.expand_dims(input_test_data[index], 0)
            image_feature = model.predict(image)
            test_features.append(image_feature)
        test_features = np.array(test_features).astype("float32")
        test_features = np.reshape(test_features, (number_of_images, output_shape[2]))

        #Save the extracted features
        print_info("Saving testing features")
        telegramSendMessage("Saving testing features")
        np.save(os.path.join(extraction_datapath,"input_testing_data_"+args.pooling+"_"+fold_name), test_features)

        #   Forcefully delete input datas from memory
        del input_train_data
        del input_test_data
        del train_features
        del test_features
        gc.collect()

    print_info("Extraction script end")
    telegramSendMessage("Extraction script end")

if __name__ == "__main__":
    try:
        extract_folds_features(sys.argv)

    except Exception as e:
        print_error('An error has occurred')
        print_error(str(e))
        telegramSendMessage('[ERROR]: An error has occurred')
        telegramSendMessage(str(e))

    exit(0)
