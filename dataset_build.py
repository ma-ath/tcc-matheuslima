import pickle
import numpy as np
from include.telegram_logger import *
from include.globals_and_functions import *
import gc

def calculate_dataset_statistics(data_mean, data_std, data_size):
    """
        This function is used to calculate mean and standart deviantion of the dataset
        without loading the full dataset into memory, as it would be loaded astype float64


        The way used to calculate mean and std was:
            1 - instead of calculating mean and std for all videos in the dataset, 
            calculate mean and std for each video in the dataset

            2 - After this, calculate the joined mean and std of all videos, by
                2.1 -  combined_mean = weighted average of all means
                2.2 -  combined_std using the formula on https://www.statstodo.com/CombineMeansSDs_Pgm.php
    """
    #   If there is only on video, there is no meaning in combining multiple statistics
    if len(data_size) == 1:
        return data_mean, data_std

    combined_mean = np.average(data_mean, axis=(0), weights=data_size)

    tn = np.sum(data_size)

    tx = np.array(data_mean)
    for i in range(data_mean.shape[0]):
        tx[i] = tx[i] * data_size[i]
    tx = np.sum(tx, axis=0)

    txx = np.square(data_std)
    for i in range(data_std.shape[0]):
        txx[i] = txx[i] * data_size[i]-1
        A = (np.square(data_mean)[i] * data_size[i])
        txx[i] = txx[i] + A
    txx = np.sum(txx, axis=0)

    combined_std = np.sqrt((txx - (np.square(tx) / tn))/(tn))

    return combined_mean, combined_std

def normalize_dataset(data, mean, std):
    """
        Reshape dataset from (n,224*224*3) to (n,224,224,3)
        Normalize dataset by subtracting mean and dividing by std
    """
    data = np.reshape(data, (data.shape[0],)+CONST_VEC_DATASET_OUTPUT_IMAGE_SHAPE).astype("float32")

    data[:, :, :, 0] -= (mean[0])
    data[:, :, :, 1] -= (mean[1])
    data[:, :, :, 2] -= (mean[2])

    data[:, :, :, 0] /= (std[0])
    data[:, :, :, 1] /= (std[1])
    data[:, :, :, 2] /= (std[2])

    return data

def dataset_build(train_videos, test_videos):
    try:
        #   --------------  Error checking

        # load config file with all processed videos in disk

        try:    #load the config file
            with open(os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_CONFIG_FILENAME), "rb") as fp:
                processed_videos = pickle.load(fp)

        except:     #   new dataset, config file does not exist
            print_error("There is no config file on system. You should run the dataset_process.py script before atempting to build it")
            exit(1)

        #   Check if i'm passing a video that is not processed in the dataset
        unknown_videos = [item for item in train_videos if item not in processed_videos]
        if unknown_videos != []:
            print_error("The following videos were not processed in the dataset:")
            for unknown_video in unknown_videos:
                print('\t'+unknown_video)
            print_error("Please run dataset_process.py before atempting to build")
            exit(1)

        unknown_videos = [item for item in test_videos if item not in processed_videos]
        if unknown_videos != []:
            print_error("The following videos were not processed in the dataset:")
            for unknown_video in unknown_videos:
                print('\t'+unknown_video)
            print_error("Please run dataset_process.py before atempting to build")
            exit(1)

        #   Check if a video is beeing used for both test and train dataset
        common_videos = [item for item in test_videos if item in train_videos]
        if common_videos != []:
            print_error("The following videos are being used for both train and test dataset. You should not train and test on same videos!")

            for common_video in common_videos:
                print('\t'+common_video)
            print_error("Please reevaluate your data")
            exit(1)

        #   --------------  Build starting
        #   This Loop:
        #
        #       get all number of frames:   train_number_of_frames
        #       get all means and std:      train_mean, train_std
        #       get all frames for dataset: input_train_data
        #       get all outputs on dataset: output_train_data
        #
        #       Test and Train codes are the same, just duplicated
        #
        #   --------------   Train dataset
        #

        print_info("Starting building process for train dataset")
        telegramSendMessage("Starting building process for train dataset")

        train_number_of_frames = []     #   Vector with the total number of frames in each video. This is necessary to calculate a number of things such as total mean, std, data loading, etc...
        first_video = True
        for video_name in train_videos:

            video_datapath = os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,video_name.replace(".", ""))

            #   Get total number of frames in video
            with open(os.path.join(video_datapath,CONST_STR_DATASET_NMB_OF_FRAMES_FILENAME), "rb") as fp:
                number_of_frames = pickle.load(fp)
            train_number_of_frames.append(number_of_frames)

            #   Get the statistics in the video
            #   Put those in the train_mean and train_std ndarrays
            with open(os.path.join(video_datapath,CONS_STR_DATASET_STATISTICS_FILENAME), "rb") as fp:
                video_statistics = pickle.load(fp)
            
            video_statistics = np.asarray(video_statistics)
            if first_video:
                train_mean = video_statistics[0]
                train_std = video_statistics[1]
            else:
                train_mean = np.vstack((train_mean, video_statistics[0]))
                train_std = np.vstack((train_std, video_statistics[1]))
       
            #   Load numpy video data. 
            video_data = np.load(os.path.join(video_datapath,CONS_STR_DATASET_STACKED_FRAMES_FILENAME))

            if first_video:
                #   Declare the train data. this is the input data used to train the model
                input_train_data = video_data
            else:
                input_train_data = np.vstack((input_train_data, video_data))

            #   Load numpy output data. 
            audio_data = np.load(os.path.join(video_datapath,CONS_STR_DATASET_AUDIODATA_FILENAME))

            # If audio file is Stereo, I take the mean between both chanels and concatenate in one channel
            try:
                if audio_data.shape[1] == 2:
                    print_info("Audio file from video "+video_name+" is stereo. Taking the average of both channels as output")
                    audio_data = np.mean(audio_data, axis=1)
            except:
                pass

            if first_video:
                #   Declare the train data. this is the input data used to train the model
                output_train_data = audio_data
            else:
                output_train_data = np.concatenate((output_train_data, audio_data))

            #   Last of all, switch of the "first video" variable
            if first_video:
                first_video = False

        #   Calculate mean and std for all dataset:
        #   Dataset preprossesing. We normalize the dataset by subtracting mean and dividing by std
        #

        train_combined_mean, train_combined_std = calculate_dataset_statistics(train_mean, train_std, train_number_of_frames)

        input_train_data = normalize_dataset(input_train_data, train_combined_mean, train_combined_std)

        # For some reason, axis 3 (colour) is fliped
        input_train_data = np.flip(input_train_data, axis=3)

        #
        #   --------------   Test dataset
        #

        test_number_of_frames = []     #   Vector with the total number of frames in each video. This is necessary to calculate a number of things such as total mean, std, data loading, etc...
        first_video = True
        for video_name in test_videos:

            video_datapath = os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,video_name.replace(".", ""))

            #   Get total number of frames in video
            with open(os.path.join(video_datapath,CONST_STR_DATASET_NMB_OF_FRAMES_FILENAME), "rb") as fp:
                number_of_frames = pickle.load(fp)
            test_number_of_frames.append(number_of_frames)

            #   Get the statistics in the video
            #   Put those in the train_mean and train_std ndarrays
            with open(os.path.join(video_datapath,CONS_STR_DATASET_STATISTICS_FILENAME), "rb") as fp:
                video_statistics = pickle.load(fp)
            
            video_statistics = np.asarray(video_statistics)
            if first_video:
                test_mean = video_statistics[0]
                test_std = video_statistics[1]
            else:
                test_mean = np.vstack((test_mean, video_statistics[0]))
                test_std = np.vstack((test_std, video_statistics[1]))
       
            #   Load numpy video data. 
            video_data = np.load(os.path.join(video_datapath,CONS_STR_DATASET_STACKED_FRAMES_FILENAME))

            if first_video:
                #   Declare the train data. this is the input data used to train the model
                input_test_data = video_data
            else:
                input_test_data = np.vstack((input_test_data, video_data))

            #   Load numpy output data.
            audio_data = np.load(os.path.join(video_datapath,CONS_STR_DATASET_AUDIODATA_FILENAME))

            # If audio file is Stereo, I take the mean between both chanels and concatenate in one channel
            try:
                if audio_data.shape[1] == 2:
                    print_info("Audio file from video "+video_name+" is stereo. Taking the average of both channels as output")
                    audio_data = np.mean(audio_data, axis=1)
            except:
                pass

            if first_video:
                #   Declare the train data. this is the input data used to train the model
                output_test_data = audio_data
            else:
                output_test_data = np.concatenate((output_test_data, audio_data))

            #   Last of all, switch of the "first video" variable
            if first_video:
                first_video = False

        #   Calculate mean and std for all dataset:
        #   Dataset preprossesing. We normalize the dataset by subtracting mean and dividing by std
        #
        test_combined_mean, test_combined_std = calculate_dataset_statistics(test_mean, test_std, test_number_of_frames)

        input_test_data = normalize_dataset(input_test_data, test_combined_mean, test_combined_std)

        # For some reason, axis 3 (colour) is fliped
        input_test_data = np.flip(input_test_data, axis=3)

        #   Return the builded dataset and the vectors with train and test video sizes
        
        print_info("Dataset built completed")
        return input_train_data, output_train_data, input_test_data, output_test_data, train_number_of_frames, test_number_of_frames

    except Exception as e:
        print_error('An error has occurred')
        print_error(str(e))
        telegramSendMessage('[ERROR]: An error has occurred')
        telegramSendMessage(str(e))

if __name__ == "__main__":
    #
    #   Script for generation of the "folds", so that training process is made upon various different scenarios
    #
    from folds import *

    #   Make directory to hold this folder information
    try:
        fold_path = os.path.join(CONST_STR_DATASET_BASE_PATH,CONST_STR_DATASET_DATAPATH,CONST_STR_DATASET_FOLDS_DATAPATH)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
    except OSError:
        print_error("Could not make directory for folds")
        telegramSendMessage("Error: Creating directory")
        exit()


    for fold in folds:
        telegramSendMessage("Creating fold "+str(fold['number']))
        print_info("Creating fold "+str(fold['number']))

        [
            input_train_data,
            output_train_data,
            input_test_data,
            output_test_data,
            train_number_of_frames,
            test_number_of_frames
        ] = dataset_build(fold["training_videos"], fold["testing_videos"])

        #   Save this dataset to the corresponding fold path
        telegramSendMessage("Saving fold "+str(fold['number'])+" to disk")
        print_info("Saving fold "+str(fold['number'])+" to disk")

        np.save(os.path.join(fold_path,"input_training_data_"+fold['name']), input_train_data)
        np.save(os.path.join(fold_path,"output_training_data_"+fold['name']), output_train_data)
        np.save(os.path.join(fold_path,"input_testing_data_"+fold['name']), input_test_data)
        np.save(os.path.join(fold_path,"output_testing_data_"+fold['name']), output_test_data)
        np.save(os.path.join(fold_path,"nof_train_"+fold['name']), train_number_of_frames)
        np.save(os.path.join(fold_path,"nof_test_"+fold['name']), test_number_of_frames)

        #   Forcefully collect all garbage in memory 
        del input_train_data
        del output_train_data
        del input_test_data
        del output_test_data
        del train_number_of_frames
        del test_number_of_frames
        gc.collect()

    telegramSendMessage("Script ended sucessfully")
    print_info("Script ended sucessfully")
