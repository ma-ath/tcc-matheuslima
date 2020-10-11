import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

"""
    This a simple script to test some LSTM networks
"""

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


real_train = np.load("lstm_analysis/real-train.npy")
pred_train = np.load("lstm_analysis/pred-train.npy")
real_test = np.load("lstm_analysis/real-test.npy")
pred_test = np.load("lstm_analysis/pred-test.npy")

training_nof = np.load("lstm_analysis/nof-train.npy")
testing_nof = np.load("lstm_analysis/nof-test.npy")

def create_dataset(X, Y, nof, time_steps=1, target_size=None):

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
            
            X_out.append(np.reshape(X[indices], (time_steps, 1)))
            Y_out.append(Y[j-target_size])

        frame_sum += nof[i]

    return np.array(X_out), np.array(Y_out)

X_train, Y_train = create_dataset(pred_train, real_train, training_nof, time_steps=100)

X_test, Y_test = create_dataset(pred_test, real_test, testing_nof, time_steps=100)

#   LSTM Model
print(X_train.shape)
input_shape = (X_train.shape[1],X_train.shape[2])
model = keras.Sequential()
model.add(keras.layers.LSTM(
  units=32, input_shape=input_shape)
    )

model.add(keras.layers.Dense(units=1))

model.compile(
  loss='mean_squared_error',
  optimizer=keras.optimizers.Adam(0.001)
)

model.summary()

history = model.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, Y_test),
    verbose=1,
    shuffle=False
)

Y_predicted = model.predict(X_test, batch_size=32)
PLOT_SIZE = Y_predicted.shape[0]*Y_predicted.shape[1]
newshape = (PLOT_SIZE, 1)
Y_predicted = np.reshape(Y_predicted, newshape)
Y_vtest = np.reshape(Y_test, newshape)

np.save('lstm_analysis'+'/lstm_real.npy', Y_vtest[0:PLOT_SIZE])
np.save('lstm_analysis'+'/lstm_pred.npy', Y_predicted)

plotAudioPowerWithPrediction(Y_vtest, Y_predicted, to_file=True, image_path='lstm_analysis', image_name='/prediction_Test_checkpoint.png')

plotTrainingLoss(history, to_file=True, image_path='lstm_analysis')