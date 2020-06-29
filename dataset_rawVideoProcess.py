import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import scipy.io.wavfile
import matplotlib.pyplot as plt
from math import floor, log10, log
import re
import pickle
from include.telegram_logger import *
from include.globals_and_functions import *

#check if needed programs are installed
if not is_tool("ffmpeg"):
    print("Please install ffmpeg for the video processing")
    exit()
    pass
if not is_tool("ffprobe"):
    print("Please install ffmpeg for the video processing")
    exit()
    pass

#Get the path of all videos
video_train_raw_datapath = [f for f in listdir(dataset_train_raw) if isfile(join(dataset_train_raw, f))]
video_test_raw_datapath = [f for f in listdir(dataset_test_raw) if isfile(join(dataset_test_raw, f))]

# # First, check what videos were already processed. To do this, we load the "config" file in the dataset directory,
# # and check for which videos are "new" in the raw folder. We only need to process those

try:    #load the config file. There are 2 config files, one for test data and one for train data
    with open(dataset_datapath+"config-train","rb") as fp:
        dataset_config_train = pickle.load(fp)
    with open(dataset_datapath+"config-test","rb") as fp:
        dataset_config_test = pickle.load(fp)

    video_train_raw_datapath = [item for item in video_train_raw_datapath if item not in dataset_config_train]
    video_test_raw_datapath = [item for item in video_test_raw_datapath if item not in dataset_config_test]
    dataset_config_train += video_train_raw_datapath
    dataset_config_test += video_test_raw_datapath
except:     #new dataset, config file does not exist
    dataset_config_train = video_train_raw_datapath
    dataset_config_test = video_test_raw_datapath

# If video_raw_datapath is empty, there are no new videos to be processed
if not video_train_raw_datapath:
    print("Train dataset is already up to date")
    telegramSendMessage('Train dataset is already up to date')
if not video_test_raw_datapath:
    print("Test dataset is already up to date")
    telegramSendMessage('Test dataset is already up to date')

telegramSendMessage('Extraction information from data!')
#extract frames and sound for each video in train data!
for video_train_raw_name in video_train_raw_datapath:
    video_train_raw_datapath = dataset_train_raw + video_train_raw_name

    datapath = dataset_train_datapath + video_train_raw_name.replace(".","")

    try:
        if not os.path.exists(dataset_train_datapath):
            os.makedirs(dataset_train_datapath)
    except OSError:
        print ('Error: Creating directory for train data')
        telegramSendMessage('Error: Creating directory for train data')
        exit ()

    try:
        if not os.path.exists(datapath):
            os.makedirs(datapath)
    except OSError:
        print ('Error: Creating directory for train data (datapath)')
        telegramSendMessage('Error: Creating directory for train data (datapath)')
        exit ()

    os_command = "ffmpeg -i "+video_train_raw_datapath+" -s "+output_resolution+" -c:a copy dataset/train/resized-"+video_train_raw_name
    os.system(os_command)

    # Extact frames from video
    cap = cv2.VideoCapture("dataset/train/resized-"+video_train_raw_name)
    currentFrame = 0    # This variable counts the actual frame extracted
    videoFrame = 0      # This variable counts the actual frame in the video
                        # I use those 2 variables so that I can change the fps of extraction
    total_number_of_video_frames = 0
    print("Extracting the frames from the video...")
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (videoFrame%frameJump == 0 ):
            if ret == True:
                # Saves image of the current frame in jpg file
                name = './'+datapath+'/'+ str(currentFrame) + '.jpg'
                #print ('Creating...' + name)
                cv2.imwrite(name,frame)
                # To stop duplicate images
                currentFrame += 1
            else:
                break

        videoFrame+=1
        total_number_of_video_frames = currentFrame

    #   After all frames has been extracted, we have to make sure that the number of frames
    #   extracted is a multiple of timeStepArray[3], so that frames from diferent videos
    #   dont mix up on the the same batch of the training process. This way we can train
    #   different videos on the same network

    number_of_extra_extracted_frames = total_number_of_video_frames % timeStepArray[2]
    print(str(total_number_of_video_frames)+" frames were extracted")
    print('number_of_extra_extracted_frames: '+str(number_of_extra_extracted_frames))

    #   We here delete those 'extra' video frames
    for i in range(total_number_of_video_frames-number_of_extra_extracted_frames,total_number_of_video_frames,1):
        extra_frame = './'+datapath+'/'+ str(i) + '.jpg'
        os.system('rm -f '+extra_frame)

    total_number_of_video_frames = total_number_of_video_frames - number_of_extra_extracted_frames

    # Save the number of frames in this video on the frames folder
    with open('./'+datapath+'/'+number_of_frames_filename,"wb") as fp:
        pickle.dump(total_number_of_video_frames, fp)
    
    print(str(total_number_of_video_frames)+" frames were extracted")
    # When everything done, release the capture
    cap.release()


    # Informations from the video
    # print("Informations from the resized video:")
    # os_command = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration,bit_rate -of default=noprint_wrappers=1 dataset/resized-"+video_name
    # os.system(os_command)
    # bit_rate = input("Please write the bitrate: ")

    #Extract the audio file from video
    audio_name = datapath+"/audioData.wav"
    os_command = "ffmpeg -i "+video_train_raw_datapath+" "+audio_name # " -f wav -ar 48000 -ab "+bit_rate+" -vn "
    os.system(os_command)

    #   Get samples from audio file generated
    print("Reading audio file ...")
    FSample, samples = scipy.io.wavfile.read(audio_name)
    samples = np.array(samples)

    #   Calculate the total power for each frame
    M = floor(samples.shape[0]/total_number_of_video_frames)    #Number of Samples used for each frame calculatiom
    St = np.zeros((total_number_of_video_frames,2))             #Array of audio power in each frame

    print("Calculating audio power for each frame ...")
    for i in range(0,total_number_of_video_frames):
        partialSumRight = 0
        partialSumLeft = 0
        for j in range(0,M):
            partialSumLeft += (1/M)*((samples[j+i*M,0])**2)
            partialSumRight += (1/M)*((samples[j+i*M,1])**2)
            pass
        if partialSumLeft > 0:
            St[i,0] = log(partialSumLeft)
        if partialSumRight > 0:
            St[i,1] = log(partialSumRight)
        pass

    # save numpy array as .npy file
    np.save(datapath+'/audioPower.npy',St)

#extract frames and sound for each video in test data!
for video_test_raw_name in video_test_raw_datapath:
    video_test_raw_datapath = dataset_test_raw + video_test_raw_name

    datapath = dataset_test_datapath + video_test_raw_name.replace(".","")

    # Creates directory to hold the train and test datas
    try:
        if not os.path.exists(dataset_test_datapath):
            os.makedirs(dataset_test_datapath)
    except OSError:
        print ('Error: Creating directory for test data')
        telegramSendMessage('Error: Creating directory for test data')
        exit ()

    try:
        if not os.path.exists(datapath):
            os.makedirs(datapath)
    except OSError:
        print ('Error: Creating directory for train data (datapath)')
        telegramSendMessage('Error: Creating directory for train data (datapath)')
        exit ()

    os_command = "ffmpeg -i "+video_test_raw_datapath+" -s "+output_resolution+" -c:a copy dataset/test/resized-"+video_test_raw_name
    os.system(os_command)

    # Extact frames from video
    cap = cv2.VideoCapture("dataset/test/resized-"+video_test_raw_name)
    currentFrame = 0    # This variable counts the actual frame extracted
    videoFrame = 0      # This variable counts the actual frame in the video
                        # I use those 2 variables so that I can change the fps of extraction
    total_number_of_video_frames = 0
    print("Extracting the frames from the video...")
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (videoFrame%frameJump == 0 ):
            if ret == True:
                # Saves image of the current frame in jpg file
                name = './'+datapath+'/'+ str(currentFrame) + '.jpg'
                #print ('Creating...' + name)
                cv2.imwrite(name,frame)
                # To stop duplicate images
                currentFrame += 1
            else:
                break

        videoFrame+=1
        total_number_of_video_frames = currentFrame

    #   After all frames has been extracted, we have to make sure that the number of frames
    #   extracted is a multiple of timeStepArray[3], so that frames from diferent videos
    #   dont mix up on the the same batch of the training process. This way we can train
    #   different videos on the same network

    number_of_extra_extracted_frames = total_number_of_video_frames % timeStepArray[2]
    print(str(total_number_of_video_frames)+" frames were extracted")
    print('number_of_extra_extracted_frames: '+str(number_of_extra_extracted_frames))

    #   We here delete those 'extra' video frames
    for i in range(total_number_of_video_frames-number_of_extra_extracted_frames,total_number_of_video_frames,1):
        extra_frame = './'+datapath+'/'+ str(i) + '.jpg'
        os.system('rm -f '+extra_frame)

    total_number_of_video_frames = total_number_of_video_frames - number_of_extra_extracted_frames

    # Save the number of frames in this video on the frames folder
    with open('./'+datapath+'/'+number_of_frames_filename,"wb") as fp:
        pickle.dump(total_number_of_video_frames, fp)

    print(str(total_number_of_video_frames)+" frames were extracted")
    # When everything done, release the capture
    cap.release()


    # Informations from the video
    # print("Informations from the resized video:")
    # os_command = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration,bit_rate -of default=noprint_wrappers=1 dataset/resized-"+video_name
    # os.system(os_command)
    # bit_rate = input("Please write the bitrate: ")

    #Extract the audio file from video
    audio_name = datapath+"/audioData.wav"
    os_command = "ffmpeg -i "+video_test_raw_datapath+" "+audio_name # " -f wav -ar 48000 -ab "+bit_rate+" -vn "
    os.system(os_command)

    #   Get samples from audio file generated
    print("Reading audio file ...")
    FSample, samples = scipy.io.wavfile.read(audio_name)
    samples = np.array(samples)

    #   Calculate the total power for each frame
    M = floor(samples.shape[0]/total_number_of_video_frames)    #Number of Samples used for each frame calculatiom
    St = np.zeros((total_number_of_video_frames,2))             #Array of audio power in each frame

    print("Calculating audio power for each frame ...")
    for i in range(0,total_number_of_video_frames):
        partialSumRight = 0
        partialSumLeft = 0
        for j in range(0,M):
            partialSumLeft += (1/M)*((samples[j+i*M,0])**2)
            partialSumRight += (1/M)*((samples[j+i*M,1])**2)
            pass
        if partialSumLeft > 0:
            St[i,0] = log(partialSumLeft)
        if partialSumRight > 0:
            St[i,1] = log(partialSumRight)
        pass

    # save numpy array as .npy file
    np.save(datapath+'/audioPower.npy',St)

# Save the information of all videos on file 
with open(dataset_datapath+"config-train","wb") as fp:
    pickle.dump(dataset_config_train, fp)
with open(dataset_datapath+"config-test","wb") as fp:
    pickle.dump(dataset_config_test, fp)

telegramSendMessage('dataset_rawVideoProcess ended successfully')