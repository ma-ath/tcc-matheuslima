import os                                   #
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU

from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from include.globals_and_functions import *
import numpy as np

MODELO = 'model_lstm_20'

path = os.path.join('cache/'+MODELO, 'architecture.json')
print('load model: '+path)
model = model_from_json(open(os.path.join('cache/'+MODELO+'/', 'architecture.json')).read())
model.load_weights(os.path.join('cache/'+MODELO, 'model_checkpoint.hdf5'))

model.compile(optimizer='adam', loss=mean_squared_error)

model.summary()

print('loading dataset')

[
    X_train,
    Y_train,
    X_test,
    Y_test
] = loadDatasetFromCache()  #Load the dataset

print('dataset load')

Y_predicted = []
Y_vtest = Y_test

# Prepare a predictionSamples vector, in order to plot it
for i in range(X_test.shape[0]):
    X_predict = np.expand_dims(X_test[i],0)

    prediction = model.predict(X_predict)

    newshape = (9,1)

    prediction = prediction[0]

    Y_predicted.append(prediction)

Y_predicted = np.array(Y_predicted).astype("float32")

PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]

newshape = (PLOT_SIZE,1)

Y_predicted = np.reshape(Y_predicted,newshape)

Y_vtest = np.reshape(Y_test,newshape)

plotAudioPowerWithPrediction(Y_vtest,Y_predicted,to_file=True,image_path='cache/'+MODELO,image_name='/prediction_val_checkpoint.png')

# ------------------- predicte over train set ------------------- #
Y_predicted = []
Y_vtest = Y_train

# Prepare a predictionSamples vector, in order to plot it
for i in range(X_train.shape[0]):
    X_predict = np.expand_dims(X_train[i],0)

    prediction = model.predict(X_predict)

    newshape = (9,1)

    prediction = prediction[0]

    Y_predicted.append(prediction)

Y_predicted = np.array(Y_predicted).astype("float32")

PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]

newshape = (PLOT_SIZE,1)

Y_predicted = np.reshape(Y_predicted,newshape)

Y_vtest = np.reshape(Y_train,newshape)

plotAudioPowerWithPrediction(Y_vtest,Y_predicted,to_file=True,image_path='cache/'+MODELO,image_name='/prediction_train_checkpoint.png')