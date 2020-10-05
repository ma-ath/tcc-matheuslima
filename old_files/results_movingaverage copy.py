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

real = np.load('/home/math/GitHub/tcc-matheuslima/results/fold_0/model_foldtraining_18/res_real_checkpoint_test.npy')
pred = np.load('/home/math/GitHub/tcc-matheuslima/results/fold_0/model_foldtraining_18/res_prediction_checkpoint_test.npy')

m_real = moving_average(real, real.size)
m_pred = moving_average(pred, real.size)

print(str(mean_squared_error(real, pred)).replace('.', ','))

print(m_real)
print(m_pred)
plotAudioPowerWithPrediction(real, pred)
