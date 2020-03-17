import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from math import floor, log

def plotAudioPower(M,St,FSample):
    audio_length = (M*St[:,0].size)/FSample
    time = np.linspace(0., audio_length, St[:,0].size)
    plt.plot(time, St[:, 0], label="Left channel")
    plt.plot(time, St[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    pass

def plotAudio(FSample,samples):
    audio_length = samples.shape[0] / FSample
    time = np.linspace(0., audio_length, samples.shape[0])
    plt.plot(time, samples[:, 0], label="Left channel")
    plt.plot(time, samples[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    pass


audio_name = "audio-VID_20200310_201757.wav"
total_number_of_video_frames = 914-1 #Do not count frame number 1

#   Get samples from audio file generated
FSample, samples = scipy.io.wavfile.read("dataset/"+audio_name)
samples = np.array(samples)
#   Calculate the total power for each frame

M = floor(samples.shape[0]/total_number_of_video_frames)    #Number of Samples used for each frame calculatiom
St = np.zeros((total_number_of_video_frames,2))             #Array of audio power in each frame

print("Calculating Audio Power ...")
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
print("Audio Power Calculated")

plt.figure(0)
plotAudioPower(M,St,FSample)
plotAudio(FSample,samples)