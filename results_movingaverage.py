import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.metrics import mean_squared_error

#   Size of moving average window

ma_n = 200

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

audio_day = np.load('/Users/mmath/Documents/GitHub/tcc-matheuslima/results/audioPower_day.npy')
audio_night = np.load('/Users/mmath/Documents/GitHub/tcc-matheuslima/results/audioPower_night.npy')

audio_day = np.delete(audio_day, -1, axis=1)
audio_night = np.delete(audio_night, -1, axis=1)

print(audio_day.shape[0])
print(audio_night.shape[0])

for i in range(32):
    check_pred_night = np.load('results/check_night_model_math_lstm_'+str(i)+'.npy')
    check_pred_day = np.load('results/check_day_model_math_lstm_'+str(i)+'.npy')
    last_pred_night = np.load('results/last_night_model_math_lstm_'+str(i)+'.npy')
    last_pred_day = np.load('results/last_day_model_math_lstm_'+str(i)+'.npy')

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

    #print(str(mean_squared_error(avr_check_pred_day,avr_audio_day)).replace('.',','))
    #print(str(mean_squared_error(avr_check_pred_night,avr_audio_night)).replace('.',','))
    #print(str(mean_squared_error(avr_last_pred_day,avr_audio_day)).replace('.',','))
    #print(str(mean_squared_error(avr_last_pred_night,avr_audio_night)).replace('.',','))

    #print("Model math_lstm_"+str(i))
    #if i == 2:
    #    plotAudioPowerWithPrediction(avr_audio_day,avr_check_pred_day)

    #print(str(checkpoint_day_correlation[0]).replace('.',','))
    #print(str(checkpoint_night_correlation[0]).replace('.',','))
    #print(str(lastepoch_day_correlation[0]).replace('.',','))
    #print(str(lastepoch_night_correlation[0]).replace('.',','))

for i in range(10):
    check_pred_night = np.load('results/check_night_model_math_nolstm_'+str(i)+'.npy')
    check_pred_day = np.load('results/check_day_model_math_nolstm_'+str(i)+'.npy')
    last_pred_night = np.load('results/last_night_model_math_nolstm_'+str(i)+'.npy')
    last_pred_day = np.load('results/last_day_model_math_nolstm_'+str(i)+'.npy')
    
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

    if i == 0:
        plotAudioPowerWithPrediction(avr_audio_night,avr_check_pred_night)

    #print(str(mean_squared_error(avr_check_pred_day,avr_audio_day)).replace('.',','))
    #print(str(mean_squared_error(avr_check_pred_night,avr_audio_night)).replace('.',','))
    #print(str(mean_squared_error(avr_last_pred_day,avr_audio_day)).replace('.',','))
    #print(str(mean_squared_error(avr_last_pred_night,avr_audio_night)).replace('.',','))

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

    #print(str(mean_squared_error(avr_check_pred_day,avr_audio_day)).replace('.',','))
    #print(str(mean_squared_error(avr_check_pred_night,avr_audio_night)).replace('.',','))
    #print(str(mean_squared_error(avr_last_pred_day,avr_audio_day)).replace('.',','))
    #print(str(mean_squared_error(avr_last_pred_night,avr_audio_night)).replace('.',','))