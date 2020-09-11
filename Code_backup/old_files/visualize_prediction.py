import os                                   #
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error, mean_absolute_error
from keras.models import model_from_json
from network_model import networkModel
import numpy as np
from PIL import Image
import re
from matplotlib import pyplot as plt
import pandas
from random import randrange, seed
from include.auxiliaryFunctions import *
from keras.utils.vis_utils import plot_model
from include.global_constants import *

                                    #Check if the model is already in cache
if os.path.isfile(os.path.join('cache', 'architecture.json')) & os.path.isfile(os.path.join('cache', 'model_weights.h5')):
    print("[INFO] Loading cached model ...")
    model = load_model()
else:
    print("[Error] There is no cached model to load. You should run network_train and network_evaluate before trying to visualize")
    raise EnvironmentError

Y_test = np.load("visualization/visualization-real-lbl.npy")
Y_predicted = np.load("visualization/visualization-prediction-lbl.npy")

plotAudioPowerWithPrediction(Y_test,Y_predicted)

plot_model(model, to_file='visualization/model.png', show_shapes=True, show_layer_names=True)