from math import log, log10, exp
import numpy as np

# Eu calculei o logaritimo na base errada no dataset processado.
# Esse arquivo fara uma simples mudanca de base

def loadLabels(PROCESSED_DATA_FOLDER):
    training_labels = np.load(PROCESSED_DATA_FOLDER+"images_training-lbl.npy")
    testing_labels = np.load(PROCESSED_DATA_FOLDER+"images_testing-lbl.npy")

    Y_train = []
    Y_test = []

    for label in training_labels:
        Y_train.append(label)
    for label in testing_labels:
        Y_test.append(label)

    #Transform the loaded data to numpy arrays
    Y_train = np.array(Y_train).astype("float32")
    Y_test = np.array(Y_test).astype("float32")

    return Y_train,Y_test

PROCESSED_DATA_FOLDER = "processedData/"

[
    Y_train,
    Y_test
] = loadLabels(PROCESSED_DATA_FOLDER)                   #Load the dataset

# Mudanca de base:
for i in range(0,Y_train.shape[0]):
    Y_train[i,0] = log10(exp(Y_train[i,0]))
    Y_train[i,1] = log10(exp(Y_train[i,1]))

for i in range(0,Y_test.shape[0]):
    Y_test[i,0] = log10(exp(Y_test[i,0]))
    Y_test[i,1] = log10(exp(Y_test[i,1]))

np.save("train-lbl-log10",Y_train)
np.save("test-lbl-log10",Y_test)