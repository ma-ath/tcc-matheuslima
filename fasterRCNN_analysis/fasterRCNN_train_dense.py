"""
    This a simple script to test somethings with the fasterRCNN network
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras import regularizers
import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import count_nonzero

def plotAudioPowerWithPrediction(testSamples,predictedSamples,to_file=False,image_path='.',image_name='/AudioPowerWithPrediction.png'):
    plt.close('all')
    
    plt.figure("Audio Power")

    audio_length = testSamples.shape[0]
    time = np.linspace(0., 1/audio_length ,audio_length)
    plt.plot(time, testSamples, label="Test Samples")
    plt.plot(time, predictedSamples, label="Predicted Samples")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio timeline")

    if to_file == False:
        plt.show()
    else:
        plt.savefig(image_path+image_name)

    pass

def plotTrainingLoss(history, to_file=False, image_path='.'):
    plt.close('all')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if to_file == False:
        plt.show()
    else:
        plt.savefig(image_path+'/lossPlot.png')
    pass

def create_dataset(X, Y, nof, time_steps=10, target_size=None):
    if target_size is None:
        target_size = int((time_steps-1)/2)
    X_out = []
    Y_out = []

    frame_sum = 0                       # This variable keeps track of what frame in training_images is being processed now
    for i in range(len(nof)):           # For each video in training_images . . .
        start_index = frame_sum+time_steps
        end_index = frame_sum+nof[i]
        for j in range(start_index, end_index):     # For each window in this video . . .
            indices = range(j-time_steps, j)
            
            X_out.append(np.reshape(X[indices], (time_steps, X.shape[1])))
            Y_out.append(Y[j-target_size])

        frame_sum += nof[i]

    return np.array(X_out), np.array(Y_out)

def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

X1_tr = np.load("fasterRCNN_analysis/M2U00001.json.dense.npy")
X2_tr = np.load("fasterRCNN_analysis/M2U00002.json.dense.npy")
X3_tr = np.load("fasterRCNN_analysis/M2U00003.json.dense.npy")
X4_tr = np.load("fasterRCNN_analysis/M2U00005.json.dense.npy")
X5_tr = np.load("fasterRCNN_analysis/M2U00006.json.dense.npy")

Y1_tr = np.load("fasterRCNN_analysis/audioData-M2U00001.npy")
Y2_tr = np.load("fasterRCNN_analysis/audioData-M2U00002.npy")
Y3_tr = np.load("fasterRCNN_analysis/audioData-M2U00003.npy")
Y4_tr = np.load("fasterRCNN_analysis/audioData-M2U00005.npy")
Y5_tr = np.load("fasterRCNN_analysis/audioData-M2U00006.npy")

X1_te = np.load("fasterRCNN_analysis/M2U00004.json.dense.npy")
X2_te = np.load("fasterRCNN_analysis/M2U00007.json.dense.npy")
X3_te = np.load("fasterRCNN_analysis/M2U00008.json.dense.npy")
X4_te = np.load("fasterRCNN_analysis/M2U00012.json.dense.npy")
X5_te = np.load("fasterRCNN_analysis/M2U00014.json.dense.npy")

Y1_te = np.load("fasterRCNN_analysis/audioData-M2U00004.npy")
Y2_te = np.load("fasterRCNN_analysis/audioData-M2U00007.npy")
Y3_te = np.load("fasterRCNN_analysis/audioData-M2U00008.npy")
Y4_te = np.load("fasterRCNN_analysis/audioData-M2U00012.npy")
Y5_te = np.load("fasterRCNN_analysis/audioData-M2U00014.npy")

Y1_tr = np.delete(Y1_tr, 1, axis=1)
while Y1_tr.shape[0] != X1_tr.shape[0]:
    Y1_tr = np.delete(Y1_tr, 1, axis=0)
Y2_tr = np.delete(Y2_tr, 1, axis=1)
while Y2_tr.shape[0] != X2_tr.shape[0]:
    Y2_tr = np.delete(Y2_tr, 1, axis=0)
Y3_tr = np.delete(Y3_tr, 1, axis=1)
while Y3_tr.shape[0] != X3_tr.shape[0]:
    Y3_tr = np.delete(Y3_tr, 1, axis=0)
Y4_tr = np.delete(Y4_tr, 1, axis=1)
while Y4_tr.shape[0] != X4_tr.shape[0]:
    Y4_tr = np.delete(Y4_tr, 1, axis=0)
Y5_tr = np.delete(Y5_tr, 1, axis=1)
while Y5_tr.shape[0] != X5_tr.shape[0]:
    Y5_tr = np.delete(Y5_tr, 1, axis=0)
Y1_te = np.delete(Y1_te, 1, axis=1)
while Y1_te.shape[0] != X1_te.shape[0]:
    Y1_te = np.delete(Y1_te, 1, axis=0)
Y2_te = np.delete(Y2_te, 1, axis=1)
while Y2_te.shape[0] != X2_te.shape[0]:
    Y2_te = np.delete(Y2_te, 1, axis=0)
Y3_te = np.delete(Y3_te, 1, axis=1)
while Y3_te.shape[0] != X3_te.shape[0]:
    Y3_te = np.delete(Y3_te, 1, axis=0)
Y4_te = np.delete(Y4_te, 1, axis=1)
while Y4_te.shape[0] != X4_te.shape[0]:
    Y4_te = np.delete(Y4_te, 1, axis=0)
Y5_te = np.delete(Y5_te, 1, axis=1)
while Y5_te.shape[0] != X5_te.shape[0]:
    Y5_te = np.delete(Y5_te, 1, axis=0)

nof_tr = [X1_tr.shape[0], X2_tr.shape[0], X3_tr.shape[0], X4_tr.shape[0], X5_tr.shape[0]]
nof_te = [X1_te.shape[0], X2_te.shape[0], X3_te.shape[0], X4_te.shape[0], X5_te.shape[0]]

nof_tr = np.array(nof_tr)
nof_te = np.array(nof_te)

X_train = np.concatenate((X1_tr, X2_tr, X3_tr, X4_tr, X5_tr), axis=0)
Y_train = np.concatenate((Y1_tr, Y2_tr, Y3_tr, Y4_tr, Y5_tr), axis=0)

X_test = np.concatenate((X1_te, X2_te, X3_te, X4_te, X5_te), axis=0)
Y_test = np.concatenate((Y1_te, Y2_te, Y3_te, Y4_te, Y5_te), axis=0)

#X_train, Y_train = create_dataset(X_train, Y_train, nof_tr, time_steps=32)
#X_test, Y_test = create_dataset(X_test, Y_test, nof_te, time_steps=32)

"""
a = X_train
b = Y_train

X_train = X_test
Y_train = Y_test

X_test = a
Y_test = b
"""

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)
sparsity = 1.0 - count_nonzero(X_train) / X_train.size
print("X_train sparsity:", sparsity)
sparsity = 1.0 - count_nonzero(X_test) / X_test.size
print("X_test sparsity:", sparsity)

X_train -= np.mean(X_train)
X_train /= np.std(X_train)

X_test -= np.mean(X_test)
X_test /= np.std(X_test)

input_shape = X_train.shape#(X_train.shape[1], X_train.shape[2])
model = keras.Sequential()
#model.add(keras.layers.LSTM(units=32, input_shape=input_shape))
model.add(keras.layers.Dense(32, activation='tanh'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(
  loss='mean_squared_error',
  optimizer=keras.optimizers.Adam()
)

model.build(input_shape)

model.summary()

plotAudioPowerWithPrediction(Y_train, X_train[:,0], to_file=False)

history = model.fit(
    X_train, Y_train,
    epochs=16,
    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=1
)

Y_predicted = model.predict(X_test, batch_size=32)
PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]
newshape = (PLOT_SIZE, 1)
Y_predicted = np.reshape(Y_predicted, newshape)
Y_vtest = np.reshape(Y_test, newshape)

#np.save('fasterRCNN_analysis'+'/real-test.npy', Y_vtest[0:PLOT_SIZE])
#np.save('fasterRCNN_analysis'+'/pred-test.npy', Y_predicted)

plotTrainingLoss(history, to_file=False, image_path='fasterRCNN_analysis')
plotAudioPowerWithPrediction(Y_vtest, Y_predicted, to_file=False)

Y_predicted = model.predict(X_train, batch_size=32)
PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]
newshape = (PLOT_SIZE, 1)
Y_predicted = np.reshape(Y_predicted, newshape)
Y_vtest = np.reshape(Y_train, newshape)

#np.save('fasterRCNN_analysis'+'/real-train.npy', Y_vtest[0:PLOT_SIZE])
#np.save('fasterRCNN_analysis'+'/pred-train.npy', Y_predicted)

plotAudioPowerWithPrediction(Y_vtest, Y_predicted, to_file=False)
