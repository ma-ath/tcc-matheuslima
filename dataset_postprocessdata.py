from include.auxiliaryFunctions import *
from include.telegram_logger import *
"""
Esse script simplesmente carrega o ddataset em memória e salva em disco
com LSTM = True
"""
try:
    if not os.path.exists("postProcessedData"):
        os.makedirs("postProcessedData")
except OSError:
	print ('Error: Creating directory')
	exit ()

PROCESSED_DATA_FOLDER = "processedData/"
image_shape = (240,240,3)
timeSteps = 10

telegramSendMessage('Loading dataset')

[
    X_train,
    Y_train,
    X_test,
    Y_test
] = loadDataset(PROCESSED_DATA_FOLDER,image_shape,timeSteps=timeSteps,lstm=True)                   #Load

telegramSendMessage('Dataset loaded')

np.save("postProcessedData/X_train.npy",X_train)
np.save("postProcessedData/Y_train.npy",Y_train)
np.save("postProcessedData/X_test.npy",X_test)
np.save("postProcessedData/Y_test.npy",Y_test)

telegramSendMessage('Script end')