import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats

#   Size of moving average window
ma_n = 50

def plotAudioPowerWithPrediction(samples,prediction,to_file=False,image_path='.',image_name='/AudioPower.png'):
    plt.close('all')
    
    plt.figure("Audio Power")

    audio_length = samples.shape[0]
    #print(audio_length)
    time = np.linspace(0., 1340 ,audio_length)
    #print(time)
    plt.plot(time, samples, label="Samples")
    plt.plot(time, prediction, label="Prediction")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio timeline")

    if to_file == False:
        plt.show()
    else:
        plt.savefig(image_path+image_name)
    pass

def moving_average(a, n=ma_n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

audio_day = np.load('dataset/test/M2U00004MPG/audioPower.npy')
audio_night = np.load('dataset/train/M2U00007MPG/audioPower.npy')

audio_day = np.delete(audio_day, -1, axis=1)
audio_night = np.delete(audio_night, -1, axis=1)

for i in range(24):
    check_pred_night = np.load('results/check_night_model_math_lstm_nohidden_'+str(i)+'.npy')
    check_pred_day = np.load('results/check_day_model_math_lstm_nohidden_'+str(i)+'.npy')
    last_pred_night = np.load('results/last_night_model_math_lstm_nohidden_'+str(i)+'.npy')
    last_pred_day = np.load('results/last_day_model_math_lstm_nohidden_'+str(i)+'.npy')

    audio_night_slice = np.delete(audio_night,slice(0,audio_night.shape[0]-check_pred_night.shape[0]),axis=0)
    avr_audio_day = moving_average(audio_day)
    avr_audio_night = moving_average(audio_night_slice)


    avr_check_pred_day = moving_average(check_pred_day)
    avr_check_pred_night = moving_average(check_pred_night)
    avr_last_pred_day = moving_average(last_pred_day)
    avr_last_pred_night = moving_average(last_pred_night)

    checkpoint_day_correlation = stats.pearsonr(avr_check_pred_day,avr_audio_day)
    lastepoch_day_correlation = stats.pearsonr(avr_last_pred_day,avr_audio_day)
    checkpoint_night_correlation = stats.pearsonr(avr_check_pred_night,avr_audio_night)
    lastepoch_night_correlation = stats.pearsonr(avr_last_pred_night,avr_audio_night)

    print("Model math_lstm_nohidden_"+str(i))
    print("Checkpoint Day Correlation:   "+str(checkpoint_day_correlation[0]))
    print("Checkpoint Night Correlation: "+str(checkpoint_night_correlation[0]))
    print("Last epoch Day Correlation:   "+str(lastepoch_day_correlation[0]))
    print("Last epoch Night Correlation: "+str(lastepoch_night_correlation[0]))