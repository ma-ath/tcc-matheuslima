try:
    import cv2
    import numpy as np
    import os
    from os import listdir
    from os.path import isfile, join
    import scipy.io.wavfile
    from math import floor, log
    import re
    import pickle
    from imutils import paths
    from include.telegram_logger import *
    from include.globals_and_functions import *

    try:    #load the config file
        with open(CONST_STR_DATASET_DATAPATH+CONST_STR_DATASET_CONFIG_FILENAME, "rb") as fp:
            dataset_config_file = pickle.load(fp)

    except:     #   new dataset, config file does not exist
        print_error("Could not find dataset config file")
        exit()

    #Last step: Calculate mean and std for each video in dataset. Save this information in disk
    for video_name in dataset_config_file:
        print_info("Loading numpy dataset for mean and std calculation")

        #   Load numpy array
        video_datapath = CONST_STR_DATASET_DATAPATH + video_name.replace(".", "")

        video_data = np.load(video_datapath+CONS_STR_DATASET_STACKED_FRAMES_FILENAME)
        video_data = np.reshape(video_data, (video_data.shape[0],)+CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE)

        #   Calculate mean and std of video
        mean = np.mean(video_data, axis=(0,1,2)).astype(float)
        std = np.std(video_data, axis=(0,1,2)).astype(float)

        statistics = [mean, std]

        print("mean: "+'\t'+str(statistics[0]))
        print("std: " +'\t'+str(statistics[1]))

        #   Save it to a file
        with open(video_datapath+CONS_STR_DATASET_STATISTICS_FILENAME, "wb") as fp:
            pickle.dump(statistics, fp)

    # Save the information of all videos on file
    with open(CONST_STR_DATASET_DATAPATH+CONST_STR_DATASET_CONFIG_FILENAME, "wb") as fp:
        pickle.dump(dataset_config_file, fp)

    print_info("Script ended successfully")
    telegramSendMessage("Script ended successfully")

except Exception as e:
    print_error('An error has occurred')
    print_error(str(e))
    telegramSendMessage('[ERROR]: An error has occurred')
    telegramSendMessage(str(e))