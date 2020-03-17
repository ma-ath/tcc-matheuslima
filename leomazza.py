import os                                   #
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU

from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from network_model import networkModel

image_shape = (240,240,3)           #input layer receives an RGB 240x240 image
lr_list = [0.001, 0.0003, 9e-05]    #loss rate for the training process (Adam optimizer)

model = networkModel(image_shape)   #model created by Leonardo Mazza

opt = Adam(lr=lr_list)              #Adam optimizer

model.compile(optimizer=opt, loss=mean_squared_error)

model.summary()