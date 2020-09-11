try:
    """
        This script was used to extract and calculate mean and standart deviantion of the dataset
        without loading the full dataset into memory, as it is loaded astype float64


        The way used to calculate mean and std was:
            1 - instead of calculating mean and std for all videos in the dataset, 
            instead calculate mean and std for each video in the dataset

            2 - After this, calculate the joined mean and std of all videos, by
                2.1 -  combined_mean = weighted average of all means
                2.2 -  combined_std using the formula on https://www.statstodo.com/CombineMeansSDs_Pgm.php
    """

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import os                                   #
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Desativa alguns warnings a respeito da minha CPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    import numpy as np
    from include.telegram_logger import *
    from include.globals_and_functions import *
    import pickle

    #Read how many frames each video has
    with open(PROCESSED_DATA_FOLDER+video_sizes_filename_train,"rb") as fp:
        video_sizes_train = pickle.load(fp)
    with open(PROCESSED_DATA_FOLDER+video_sizes_filename_test,"rb") as fp:
        video_sizes_test = pickle.load(fp)

    video_sizes_train = np.array(video_sizes_train)
    

    #print('video_sizes_train: '+str(video_sizes_train))
    #print('loading dataset. . .')
    #X = np.load(PROCESSED_DATA_FOLDER+"images_training-img.npy")
    #X = np.reshape(X,(X.shape[0],)+image_shape)
    #print('calculating mean and std for training_set. . .')

    # -------------------- CALCULO DIRETO -------------------- #
    #mean = np.mean(X,axis=(0,1,2))
    #std = np.std(X,axis=(0,1,2))

    #print('mean calculado sem separar: '+str(mean))
    #print('std calculado sem separar: '+str(std))

    # -------------------- CALCULO DIRETO -------------------- #

    # -------------------- CALCULO POR PARTES -------------------- #
    #i = 0
    #mean_train = []
    #std_train = []
    #for video_size in video_sizes_train:
    #    mean = np.mean(X[i:i+video_size],axis=(0,1,2))
    #    std = np.std(X[i:i+video_size],axis=(0,1,2))

    #    mean_train.append(mean)
    #    std_train.append(std)

    #    i+=video_size

    #mean_train = np.array(mean_train).astype("float64")
    #std_train = np.array(std_train).astype("float64")

    mean_train = np.load('mean_train',allow_pickle=True)
    std_train = np.load('std_train',allow_pickle=True)

    print('mean_train:\n'+str(mean_train))
    print('std_train:\n'+str(std_train))

    combined_mean = np.average(mean_train,axis=(0),weights=video_sizes_train)

    tn = np.sum(video_sizes_train)
    #print('tn: '+str(tn))
    tx = np.array(mean_train)
    for i in range(mean_train.shape[0]):
        tx[i] = tx[i] * video_sizes_train[i]
    tx = np.sum(tx,axis=0)
    #print('tx: '+str(tx))

    txx = np.square(std_train)
    for i in range(std_train.shape[0]):
        txx[i] = txx[i] * video_sizes_train[i]-1
        A = (np.square(mean_train)[i] * video_sizes_train[i])
        txx[i] = txx[i] + A
    txx = np.sum(txx,axis=0)
    #print('txx: '+str(txx))

    combined_std = np.sqrt((txx - ( np.square(tx) / tn ))/(tn))

    print('mean calculado a partir de mean_train: '+str(combined_mean))
    print('std calculado a partir de std_train: '+str(combined_std))

    # -------------------- CALCULO POR PARTES -------------------- #
except Exception as e:
    print('an error has occurred')
    print(str(e))