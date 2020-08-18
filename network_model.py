import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Conv3D, GlobalAveragePooling2D, GlobalMaxPooling2D,Dense, Input, Flatten, ConvLSTM2D, LSTM, TimeDistributed, BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from tensorflow.keras import regularizers
from include.globals_and_functions import *

def networkModel(network):

    if network['features_input'] == False:
        inputs = Input(shape=(network['time_steps'],)+image_shape)

        convolutional_layer = VGG16(weights='imagenet', include_top=False,input_shape=image_shape)
        for layer in convolutional_layer.layers[:]:
            layer.trainable = False     #Freezes all layers in the vgg16
    
        vgg16_time = TimeDistributed(convolutional_layer,name='VGG16')(inputs)
    else:
        if network['pooling_input'] == None:
            inputs = Input(shape=(network['time_steps'],)+VGG16_OUTPUT_SHAPE)         
        elif network['pooling_input'] == 'GAP' or 'GMP':
            inputs = Input(shape=(network['time_steps'],VGG16_OUTPUT_SHAPE[2]))
        vgg16_time = inputs

    if network['rcnn_type'] == 'convlstm':
        rcnn = ConvLSTM2D(return_sequences = True,
                        kernel_size=network['rcnn_kernel'],
                        filters=network['rcnn_filters'],
                        padding='valid',
                        data_format='channels_last',
                        activation=network['rcnn_activation'])(vgg16_time)

        if network['pooling'] == 'GAP':
            POOLING = TimeDistributed(GlobalAveragePooling2D(data_format=None),name='GAP')(rcnn)
        elif network['pooling'] == 'GMP':
            POOLING = TimeDistributed(GlobalMaxPooling2D(data_format=None),name='GMP')(rcnn)

        FC = TimeDistributed(Dense(network['fc_nlinear_size'],
                                    activation=network['fc_nlinear_activation'],
                                    name='dense_nlinear'),
                                    name='FC_nonlinear')(POOLING)

        outputs = TimeDistributed(Dense(1, activation='linear', name='dense_1'),name='FC_linear')(FC)

    elif network['rcnn_type'] == 'lstm':
        #
        #   This LSTM Model is based on the Paper "Quo Vadis, action recognition? A new model and the kinetics dataset"
        #
        if network['pooling_input'] == None:
            if network['pooling'] == 'GAP':
                POOLING = TimeDistributed(GlobalAveragePooling2D(data_format=None),name='GAP')(vgg16_time)
            elif network['pooling'] == 'GMP':
                POOLING = TimeDistributed(GlobalMaxPooling2D(data_format=None),name='GMP')(vgg16_time)

            rcnn = LSTM(network['lstm_units'],dropout=network['lstm_dropout'])(POOLING)#(normalization)       

        else:
            rcnn = LSTM(network['lstm_units'],dropout=network['lstm_dropout'])(vgg16_time)   

        if network['hidden_fc'] == True:
            FC = Dense(network['fc_nlinear_size'],
                        activation=network['fc_nlinear_activation'],
                        name='dense_nlinear',activity_regularizer=network['fc_nlinear_activity_regularizer'])(rcnn)

            if network['overlaping_window'] == False:
                outputs = Dense(network['time_steps'], activation='linear', name='dense_out')(FC)
            else:
                outputs = Dense(1, activation='linear', name='dense_out')(FC)
        else:
            if network['overlaping_window'] == False:
                outputs = Dense(network['time_steps'], activation='linear', name='dense_out')(rcnn)
            else:
                outputs = Dense(1, activation='linear', name='dense_out')(rcnn)

    elif network['rcnn_type'] == 'no_rnn':

        if network['features_input'] == False:
            inputs = Input(shape=image_shape)

            convolutional_layer = VGG16(weights='imagenet', include_top=False,input_shape=image_shape)
            for layer in convolutional_layer.layers[:]:
                layer.trainable = False     #Freezes all layers in the vgg16
        
            vgg16_time = convolutional_layer(inputs)
        else:
            inputs = Input(shape=VGG16_OUTPUT_SHAPE)
            vgg16_time = inputs

        if network['pooling'] == 'GAP':
            POOLING = GlobalAveragePooling2D(data_format=None)(vgg16_time)
        elif network['pooling'] == 'GMP':
            POOLING = GlobalMaxPooling2D(data_format=None)(vgg16_time)
    
        if network['hidden_fc'] == True:
            FC = Dense(network['fc_nlinear_size'],
                        activation=network['fc_nlinear_activation'],
                        name='dense_nlinear',activity_regularizer=network['fc_nlinear_activity_regularizer'])(POOLING)

            outputs = Dense(1, activation='linear', name='dense_out')(FC)
        else:
            outputs = Dense(1, activation='linear', name='dense_out')(POOLING)


    model = Model(inputs=inputs, outputs=outputs)

    return model
    #Colocar uma time distributed na camada FC
    #Olhar a ResNet/colocar um bypass direto da entrada pra saída da lstm
    #Fazer a LSTM funcionar

if __name__ == "__main__":
    from keras.optimizers import Adam
    from keras.losses import mean_squared_error
    from keras.utils.vis_utils import plot_model
    from networks import *

    for network in networks:

        model = networkModel(network)

        #Training Optimizer
        opt = network['optimizer']

        #Loss function to minimize
        if network['loss_function'] == 'mse':
            loss_function = mean_squared_error
        else:
            print("[Warning]: loss function does not suported. Using default (mse)")
            loss_function = mean_squared_error

        model.compile(optimizer=opt, loss=loss_function) #, metrics=['accuracy'])  #We can not use accuracy as a metric in this model

        #Show network model in terminal
        print('Fitting the following model:')
        for key, value in network.items():
            print(key, ' : ', value)
        model.summary()
        plot_model(model,show_shapes=True, show_layer_names=True)
      
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
    ##############################################################################

    # input_layer=Input(shape=image_shape)
    #   the vgg16 model is initialized with the imagenet's weights.
    #   the fully connected layers present in the vgg16 by default are 
    #   not added to the model, as we are going to add our own FC layer
    #   for the classification task. This is made by setting include_top=False  
    #
    #   Also, this layer is not trainable. we freeze all convolutional layers with
    #   the following for loop

    convolutional_layer = VGG16(weights='imagenet', include_top=False,input_shape=image_shape)

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