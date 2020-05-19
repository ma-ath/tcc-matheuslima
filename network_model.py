import keras
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Input, Flatten
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential

#############################################################################
#                           Model Generation
#   Model Created by Leonardo Mazza, in his master thesis
#   Here, i am reproducing model #35, the one with the lowest training loss
#   Model consists of:
#   1º - VGG16 convolutional input
#   2º - GAP pooling layer
#   3º - 1 Fully Connected layer (128)
#   4º - 1 Fully Connected output layer (2)
# 
# Optmization is made with an Adam optimizer, learning_schedule = [0.001, 0.0003, 9e-05] 
##############################################################################

def networkModel(image_shape):
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
    FC_layer = Dense(128, activation='linear', name='dense_128')#(GAP_layer)
    model_layer.add(FC_layer)
        #NOTE:
        #ORIGINALY, THE LAST OUTPUT LAYER IS MADE OF ONLY ONE SINGLE RELU NEURON. I
        #DECIDED TO PUT TWO OUTPUTS, ONE FOR EACH AUDIO CHANNEL
    output_layer = Dense(1, activation='linear', name='dense_1')#(FC_layer)
    model_layer.add(output_layer)

    #   Network Model
    #   convolutional_layer -> GAP_layer -> FC_layer(128) -> FC_layer(1)
    #   [NOTE] Using the statement Model() of keras is important to solve a bug later on, when trying to import
    # the model from a json file.
    model = Model(inputs=model_layer.input, outputs=model_layer.output)

    return model

if __name__ == "__main__":
    from keras.optimizers import Adam
    from keras.losses import mean_squared_error
    image_shape = (240,240,3)

    model = networkModel(image_shape)
    model.compile(optimizer='adam', loss=mean_squared_error)
    model.summary()                     #Show network model