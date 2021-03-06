import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras

from keras.layers import Input, LSTM, TimeDistributed, Dense, BatchNormalization, Concatenate, Flatten, Dropout
from keras.models import Model
from keras import regularizers

from include.globals_and_functions import *

def networkModel(network):
    #
    #   Function that returns the model for training
    #
    if network['lstm']:
        #
        #   This LSTM Model is based on the Paper "Quo Vadis, action recognition? A new model and the kinetics dataset"
        #
        if network['hiddenfc_before_lstm']:
            if (network['cnn'] is None):
                #   I didn't bother to program the case where a model has no cnn. You should already process it when building dataset
                print_error("Erro in declaration of model "+network['model_name'])
                print_error("Your model has to have a specified cnn layer")
                exit(1)
            #   ------- Input layer -------   #
            if (network['pooling'] is None):
                if (network['cnn'] == 'vgg16'):
                    input_shape = (network['time_steps'],)+CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE
                if (network['cnn'] == 'resnet50'):
                    input_shape = (network['time_steps'],)+CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE
                if (network['cnn'] == 'inceptionV3'):
                    input_shape = (network['time_steps'],)+CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE
            else:
                if (network['cnn'] == 'vgg16'):
                    input_shape = (network['time_steps'], CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[2])
                if (network['cnn'] == 'resnet50'):
                    input_shape = (network['time_steps'], CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[2])
                if (network['cnn'] == 'inceptionV3'):
                    input_shape = (network['time_steps'], CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[2])

            if not network['lstm_stateful']:
                layer_input = Input(shape=input_shape)
            else:
                layer_input = Input(shape=input_shape, batch_shape=(network['batch_size'],)+input_shape)

           #   ------- Hidden FC layer -------   #
            if network['hiddenfc']:
                layer_hidden_fc_before_dropout = TimeDistributed(
                            Dense(network['hiddenfc_size'],
                            activation=network['hiddenfc_activation'],
                            activity_regularizer=network['hiddenfc_activity_regularizer'])
                                )(layer_input)
                layer_hidden_fc = Dropout(network['hiddenfc_dropout'])(layer_hidden_fc_before_dropout)

                #   ------- Linear layer -------   #
                if network['overlap_windows']:
                    layer_linear_output = TimeDistributed(Dense(1, activation='linear'))(layer_hidden_fc)
                else:
                    layer_linear_output = TimeDistributed(Dense(network['time_steps'], activation='linear'))(layer_hidden_fc)
            else:
                if network['overlap_windows']:
                    layer_linear_output = TimeDistributed(Dense(1, activation='linear'))(layer_input)
                else:
                    layer_linear_output = TimeDistributed(Dense(network['time_steps'], activation='linear'))(layer_input)

            #   ------- LSTM layer -------   #
            if network['lstm_batchnormalization']:
                layer_batchnorm = BatchNormalization()(layer_linear_output)
                layer_rnn = LSTM(network['lstm_outputsize'], dropout=network['lstm_dropout'], stateful=network['lstm_stateful'])(layer_batchnorm)
            else:
                layer_rnn = LSTM(network['lstm_outputsize'], dropout=network['lstm_dropout'], stateful=network['lstm_stateful'])(layer_linear_output)
            #   ------- Output -------  #
            if network['overlap_windows']:
                layer_output = Dense(1, activation="linear")(layer_rnn)
            else:
                layer_output = Dense(network['time_steps'], activation="linear")(layer_rnn)

        else:
            if (network['cnn'] is None):
                #   I didn't bother to program the case where a model has no cnn. You should already process it when building dataset
                print_error("Erro in declaration of model "+network['model_name'])
                print_error("Your model has to have a specified cnn layer")
                exit(1)
            #   ------- Input layer -------   #
            if (network['pooling'] is None):
                if (network['cnn'] == 'vgg16'):
                    input_shape = (network['time_steps'],)+CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE
                if (network['cnn'] == 'resnet50'):
                    input_shape = (network['time_steps'],)+CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE
                if (network['cnn'] == 'inceptionV3'):
                    input_shape = (network['time_steps'],)+CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE
            else:
                if (network['cnn'] == 'vgg16'):
                    input_shape = (network['time_steps'], CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[2])
                if (network['cnn'] == 'resnet50'):
                    input_shape = (network['time_steps'], CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[2])
                if (network['cnn'] == 'inceptionV3'):
                    input_shape = (network['time_steps'], CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[2])

            if not network['lstm_stateful']:
                layer_input = Input(shape=input_shape)
            else:
                layer_input = Input(shape=input_shape, batch_shape=(network['batch_size'],)+input_shape)
            #   ------- LSTM layer -------   #
            
            layer_rnn = LSTM(network['lstm_outputsize'], dropout=network['lstm_dropout'], stateful=network['lstm_stateful'])(layer_input)

            #   ------- Hidden FC layer -------   #

            if network['hiddenfc']:

                layer_hidden_fc_before_dropout = Dense(network['hiddenfc_size'],
                            activation=network['hiddenfc_activation'],
                            activity_regularizer=network['hiddenfc_activity_regularizer'])(layer_rnn)

                layer_hidden_fc = Dropout(network['hiddenfc_dropout'])(layer_hidden_fc_before_dropout)

                #   ------- Output layer -------   #
                if network['overlap_windows']:
                    layer_output = Dense(1, activation='linear')(layer_hidden_fc)
                else:
                    layer_output = Dense(network['time_steps'], activation='linear')(layer_hidden_fc)
            else:
                if network['overlap_windows']:
                    layer_output = Dense(1, activation='linear')(layer_rnn)
                else:
                    layer_output = Dense(network['time_steps'], activation='linear')(layer_rnn)
    else:
        if (network['cnn'] == None):
            #   I didn't bother to program the case where a model has no cnn. You should already process it when building dataset
            print_error("Erro in declaration of model "+network['model_name'])
            print_error("Your model has to have a specified cnn layer")
            exit(1)

        if (network['pooling'] == None):
            if (network['cnn'] == 'vgg16'):
                layer_input = Input(shape=CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE)
            if (network['cnn'] == 'resnet50'):
                layer_input = Input(shape=CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE)
            if (network['cnn'] == 'inceptionV3'):
                layer_input = Input(shape=CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE)
        else:
            if (network['cnn'] == 'vgg16'):
                layer_input = Input(shape=[CONST_VEC_NETWORK_VGG16_OUTPUTSHAPE[2]])
            if (network['cnn'] == 'resnet50'):
                layer_input = Input(shape=[CONST_VEC_NETWORK_RESNET50_OUTPUTSHAPE[2]])
            if (network['cnn'] == 'inceptionV3'):
                layer_input = Input(shape=[CONST_VEC_NETWORK_INCEPTIONV3_OUTPUTSHAPE[2]])

        #   ------- Hidden FC layer -------   #
        if network['hiddenfc']:

            layer_hidden_fc_before_dropout = Dense(network['hiddenfc_size'],
                        activation=network['hiddenfc_activation'],
                        activity_regularizer=network['hiddenfc_activity_regularizer'])(layer_input)
            layer_hidden_fc = Dropout(network['hiddenfc_dropout'])(layer_hidden_fc_before_dropout)

            #   ------- Output layer -------   #
            layer_output = Dense(1, activation='linear')(layer_hidden_fc)
            
        else:
            layer_output = Dense(1, activation='linear')(layer_input)

    #   This is just a test. Here, we add an alternative input to the model, with the
    #   features extracted from the fasterRCNN (Pedro Cayres)
    #
    #   Basic model i made requires to use a hidden layer on the 
    if network['fasterRCNN_support']:
        if not network['hiddenfc']:
            print_error("When using fasterRCNN auxiliary input, make sure to use a hidden layer on your model")
            exit()
        
        #   Definition of input layer
        #   I know it's mendokusai
        if network['lstm']:
            if not network['lstm_stateful']:
                if network['fasterRCNN_type'] == 'dense':
                    fasterRCNN_input_shape = (None, 6)
                elif network['fasterRCNN_type'] == 'sparse':
                    fasterRCNN_input_shape = (None, 20, 8)
            else:
                if network['fasterRCNN_type'] == 'dense':
                    fasterRCNN_input_shape = (network['batch_size'], 6)
                elif network['fasterRCNN_type'] == 'sparse':
                    fasterRCNN_input_shape = (network['batch_size'], 5, 8, 2)
        else:
            if network['fasterRCNN_type'] == 'dense':
                fasterRCNN_input_shape = (None, 6)
            elif network['fasterRCNN_type'] == 'sparse':
                fasterRCNN_input_shape = (None, 20, 8)

        fasterRCNN_input = Input(batch_shape=fasterRCNN_input_shape)

        #   When using the dense input, we need a Flatten layer before the hidden layer
        if network['fasterRCNN_type'] == 'dense':
            fasterRCNN_dense = Dense(network['fasterRCNN_dense_size'], activation='tanh')(fasterRCNN_input)                    
        elif network['fasterRCNN_type'] == 'sparse':
            fasterRCNN_flat = Flatten()(fasterRCNN_input)
            fasterRCNN_dense = Dense(network['fasterRCNN_dense_size'], activation='tanh')(fasterRCNN_flat)                    

        layer_concatenate = Concatenate(axis=1)([layer_hidden_fc, fasterRCNN_dense])
        layer_kyotsu_dense = Dense(network['hiddenfc_size'], activation='tanh')(layer_concatenate)
        layer_output = Dense(1, activation='linear')(layer_kyotsu_dense)

        model = Model(inputs=[layer_input, fasterRCNN_input], outputs=layer_output)

        return model

    model = Model(inputs=layer_input, outputs=layer_output)

    return model

if __name__ == "__main__":
    from keras.optimizers import Adam
    from keras.losses import mean_squared_error
    from keras.utils.vis_utils import plot_model
    from networks import *

    for network in networks:

        model = networkModel(network)

        for key, value in network.items():
            print(key, ' : ', value)

        model.summary()

        plot_model(model, show_shapes=True, show_layer_names=True)
      
def networkModel_leomazza(image_shape):
    #############################################################################
    #                           Model Generation
    #   Model Created by Leonardo Mazza, in his master thesis
    #   Here, i am reproducing model #35, the one with the lowest training loss
    #   Model consists of:
    #   1º - VGG16 convolutional input
    #   2º - LSTM layer
    #   3º - GAP pooling layer
    #   4º - 1 Fully Connected layer (128)
    #   5º - 1 Fully Connected output layer (2)
    # 
    # Optmization is made with an Adam optimizer, learning_schedule = [0.001, 0.0003, 9e-05]
    #
    #   I DO NOT use this function anywhere in the code, and is here to documentation porposues
    #   This function is heavely un-optimized and cannot process the amount of data required in my work
    #
    ##############################################################################

    # input_layer=Input(shape=image_shape)
    #   the vgg16 model is initialized with the imagenet's weights.
    #   the fully connected layers present in the vgg16 by default are 
    #   not added to the model, as we are going to add our own FC layer
    #   for the classification task. This is made by setting include_top=False  
    #
    #   Also, this layer is not trainable. we freeze all convolutional layers with
    #   the following for loop

    from keras.applications.vgg16 import VGG16
    from keras.models import Sequential
    from keras.layers import GlobalAveragePooling2D, Dense

    convolutional_layer = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)

    model_layer = Sequential()

    for layer in convolutional_layer.layers[:]:
        layer.trainable = False     #Freezes all layers in the vgg16
        model_layer.add(layer)

    #convolutional_layer_output = convolutional_layer(input_layer)
    #   We add to the model a GAP and a FC layer
    GAP_layer = GlobalAveragePooling2D(data_format=None)#(convolutional_layer.output)
    model_layer.add(GAP_layer)
    # model.add(Flatten())
    FC_layer = Dense(128, activation='tanh', name='dense_128')#(GAP_layer)
    model_layer.add(FC_layer)
    output_layer = Dense(1, activation='linear', name='dense_1')#(FC_layer)
    model_layer.add(output_layer)

    #   Network Model
    #   convolutional_layer -> GAP_layer -> FC_layer(128) -> FC_layer(1)
    #   [NOTE] Using the statement Model() of keras is important to solve a bug later on, when trying to import
    # the model from a json file.
    model = Model(inputs=model_layer.input, outputs=model_layer.output)

    return model