import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from include.telegram_logger import *
from include.globals_and_functions import *
"""
This script simply loads the dataset and saves to disk as a numpy file.
This way, the dataset processing is done only one time 
"""
try:
    if not os.path.exists(DATASET_CACHE_FOLDER):
        os.makedirs(DATASET_CACHE_FOLDER)

    telegramSendMessage('Loading the dataset to RAM')

    [
        X_train,
        Y_train,
        X_test,
        Y_test
    ] = loadDatasetLSTM(overlap_windows=True)   #Load

    telegramSendMessage('Dataset loaded')

    np.save(DATASET_CACHE_FOLDER+"X_train.npy",X_train)
    np.save(DATASET_CACHE_FOLDER+"Y_train.npy",Y_train)
    np.save(DATASET_CACHE_FOLDER+"X_test.npy",X_test)
    np.save(DATASET_CACHE_FOLDER+"Y_test.npy",Y_test)

    telegramSendMessage('Script ended sucessfully')

except Exception as e:

    telegramSendMessage('an error has occurred')
    telegramSendMessage(str(e))