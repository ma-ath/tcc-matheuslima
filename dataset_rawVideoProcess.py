import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import scipy.io.wavfile
import matplotlib.pyplot as plt
from math import floor, log
import re
import pickle

# Resizes the video for the 240Nx240 screen resolution
output_resolution = "240x240"   #This string determines the output resolution of the resized video
frameJump = 10                  #This number determines what is the next frame in the amostration process
dataset_datapath = "./dataset/" #Datapath for the dataset
dataset_raw = "./dataset/raw/"  #Path in which all raw video files are

#Function that returns if a program is installed at the machine
def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None
#Function that plots both power and time series of an audio at once
def plotAudio(FSample,samples,M,St):
    plt.figure("Audio Information")

    plt.subplot(211)
    audio_length1 = samples.shape[0] / FSample
    time1 = np.linspace(0., audio_length1, samples.shape[0])
    plt.plot(time1, samples[:, 0], label="Left channel")
    plt.plot(time1, samples[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio timeline")

    plt.subplot(212)
    audio_length2 = (M*St[:,0].size)/FSample
    time2 = np.linspace(0., audio_length2, St[:,0].size)
    plt.plot(time2, St[:, 0], label="Left channel")
    plt.plot(time2, St[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Power stream timeline")

    plt.show()
    pass

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
video_raw_datapath = [f for f in listdir(dataset_raw) if isfile(join(dataset_raw, f))]

# First, check what videos were already processed. To do this, we load the "config" file in the dataset directory,
# and check for which videos are "new" in the raw folder. We only need to process those

try:    #load the config file
    with open(dataset_datapath+"config","rb") as fp:
        dataset_config = pickle.load(fp)

    video_raw_datapath = [item for item in video_raw_datapath if item not in dataset_config]
    dataset_config += video_raw_datapath
except:     #new dataset, config file does not exist
    dataset_config = video_raw_datapath

# If video_raw_datapath is empty, there are no new videos to be processed
if not video_raw_datapath:
    print("Dataset is already up to date")


#extract frames and sound for each video
for video_raw_name in video_raw_datapath:
    video_raw_datapath = dataset_raw + video_raw_name
    datapath = dataset_datapath + video_raw_name.replace(".","")

    try:
        if not os.path.exists(datapath):
            os.makedirs(datapath)
    except OSError:
        print ('Error: Creating directory of data')

    os_command = "ffmpeg -i "+video_raw_datapath+" -s "+output_resolution+" -c:a copy dataset/resized-"+video_raw_name
    os.system(os_command)

    # Extact frames from video
    cap = cv2.VideoCapture("dataset/resized-"+video_raw_name)
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
    os_command = "ffmpeg -i "+video_raw_datapath+" "+audio_name # " -f wav -ar 48000 -ab "+bit_rate+" -vn "
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
        St[i,0] = log(partialSumLeft)
        St[i,1] = log(partialSumRight)
        pass

    # save numpy array as .npy file
    np.save(datapath+'/audioPower.npy',St)

#    print("Audio Power Calculated")
#    print("Plotting audio information")
#    plotAudio(FSample,samples,M,St)


# Save the information of all videos on file 
with open(dataset_datapath+"config","wb") as fp:
    pickle.dump(dataset_config, fp)