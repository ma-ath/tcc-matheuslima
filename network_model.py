import keras
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.utils.vis_utils import plot_model
from keras.models import Model

#############################################################################
#                           Model Generation
#   Model Created by Leonardo Mazza, in his master thesis
#   Here, i am reproducing model #35, the one with the lowest training loss
#   Model consists of:
#   1º - VGG16 convolutional input
#   2º - GAP pooling layer
#   3º - 1 Fully Connected layer
# 
# Optmization is made with an Adam optimizer, lr_list = [0.001, 0.0003, 9e-05] 
##############################################################################

def networkModel(image_shape):
    input_layer=Input(shape=image_shape)
    #   the vgg16 model is initialized with the imagenet's weights.
    #   the fully connected layers present in the vgg16 by default are 
    #   not added to the model, as we are going to add our own FC layer
    #   for the classification task. This is made by setting include_top=False  
    #
    #   Also, this layer is not trainable
    convolutional_layer = VGG16(weights='imagenet', include_top=False,input_tensor=input_layer)
    convolutional_layer.trainable = False

    #convolutional_layer_output = convolutional_layer(input_layer)
    #   We add to the model a GAP and a FC layer
    GAP_layer = GlobalAveragePooling2D(data_format=None)(convolutional_layer.output)
    FC_layer = Dense(128, activation='relu', name='dense_128')(GAP_layer)
        #NOTE:
        #ORIGINALY, THE LAST OUTPUT LAYER IS MADE OF ONLY ONE SINGLE RELU NEURON. I
        #DECIDED TO PUT TWO OUTPUTS, ONE FOR EACH AUDIO CHANNEL
    output_layer = Dense(2, activation='relu', name='dense_1')(FC_layer)
    #   Network Model
    #   convolutional_layer -> GAP_layer -> FC_layer
    model = Model(inputs=input_layer, outputs=output_layer)
    #plot_model(model, to_file='vgg.png')
    return model