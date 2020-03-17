import cv2
import numpy as np
import os
import scipy.io.wavfile
import matplotlib.pyplot as plt
from math import floor, log

#Function that returns if a program is installed at the machine
def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None
#Function that plots the power of given audio
def plotAudioPower(M,St,FSample):
    audio_length = (M*St[:,0].size)/FSample
    time = np.linspace(0., audio_length, St[:,0].size)
    plt.plot(time, St[:, 0], label="Left channel")
    plt.plot(time, St[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Power stream timeline")
    plt.show()
    pass
#Function that plots the power as a time series for an audio
def plotAudioPower(M,St,FSample):
    audio_length = (M*St[:,0].size)/FSample
    time = np.linspace(0., audio_length, St[:,0].size)
    plt.plot(time, St[:, 0], label="Left channel")
    plt.plot(time, St[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Power stream timeline")
    plt.show()
    pass
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

# Resizes the video for the 240Nx240 screen resolution
output_resolution = "240x240"   #This string determines the output resolution of the resized video
frameJump = 1                   #This number determines what is the next frame in the amostration process

video_name = input("Please write the video name (w/ extension): ")

try:
    if not os.path.exists('dataset/images'):
        os.makedirs('dataset/images')
except OSError:
    print ('Error: Creating directory of data')

os_command = "ffmpeg -i "+video_name+" -s "+output_resolution+" -c:a copy dataset/resized-"+video_name
os.system(os_command)

# Extact frames from video
cap = cv2.VideoCapture("dataset/resized-"+video_name)
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
            name = './dataset/images/' + str(currentFrame) + '.jpg'
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
cv2.destroyAllWindows()

# Informations from the video
print("Informations from the resized video:")
os_command = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration,bit_rate -of default=noprint_wrappers=1 dataset/resized-"+video_name
os.system(os_command)
bit_rate = input("Please write the bitrate: ")
#Extract the audio file from video
audio_name = "dataset/audio-"+video_name[:len(video_name)-4]+".wav"
os_command = "ffmpeg -i "+video_name+" -f wav -ar 48000 -ab "+bit_rate+" -vn "+audio_name
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

# save numpy array as csv file
np.save('audioPower.npy',St)

print("Audio Power Calculated")
print("Plotting audio information")
plotAudio(FSample,samples,M,St)