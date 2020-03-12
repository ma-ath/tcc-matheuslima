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
    input_layer = Input(shape=image_shape,name = 'input_layer')
    #   the vgg16 model is initialized with the imagenet's weights.
    #   the fully connected layers present in the vgg16 by default are 
    #   not added to the model, as we are going to add our own FC layer
    #   for the classification task. This is made by setting include_top=False  
    convolutional_layer = VGG16(weights='imagenet', include_top=False)
    #convolutional_layer.summary()
    convolutional_layer_output = convolutional_layer(input_layer)
    #   We add to the model a GAP and a FC layer
    GAP_layer = GlobalAveragePooling2D(data_format=None)(convolutional_layer_output)
    FC_layer = Dense(1, activation='relu', name='dense_1')(GAP_layer)
    #   Network Model
    #   input_layer -> convolutional_layer -> GAP_layer -> FC_layer
    model = Model(inputs=input_layer, outputs=FC_layer)
    #plot_model(model, to_file='vgg.png')
    return model