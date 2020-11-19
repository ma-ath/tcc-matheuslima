import matplotlib.pyplot as plt
import numpy as np
import os

#   --  Nome de arquivos
RESULTS_PATH = 'results-true'
REAL_CHECK_DATANAME = 'res_real_checkpoint_test.npy'
REAL_LAST_DATANAME = 'res_real_lastepoch_test.npy'
PRED_CHECK_DATANAME = 'res_prediction_checkpoint_test.npy'
PRED_LAST_DATANAME = 'res_prediction_lastepoch_test.npy'
MODEL_BASENAME = 'model_foldtraining_'
FOLD_BASENAME = 'fold_'
NMB_OF_FOLDS = 10

#   --  Funcao q plota
def plotAudioPowerWithPrediction(testSamples,predictedSamples):
    plt.close('all')
    plt.figure("Audio Power")
    audio_length = testSamples.shape[0]
    time = np.linspace(0., 0.33333333*audio_length, audio_length)
    plt.plot(time, testSamples, label="Test Samples")
    plt.plot(time, predictedSamples, label="Predicted Samples")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Audio timeline")
    plt.show()

#   --  Moving average
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    model = input('Selecione o modelo:  ')
    fold = input("Selecione o fold (obrigatorio):  ")
    checkpoint = input("checkpoint ou last epoch? (check: 0; last: 1):   ")
    ma_n = input("plotar grafico com filtro de moving average? (0 para nao, N para valor de moving average):  ")

    if ma_n == "0" or ma_n == '':
        ma_n = "1"
    model = MODEL_BASENAME+model
    fold = FOLD_BASENAME+fold

    print("carregando model",model,"do fold",fold,"e moving average",ma_n)

    pred_check = np.load(os.path.join(RESULTS_PATH, fold, model, PRED_CHECK_DATANAME))
    real_check = np.load(os.path.join(RESULTS_PATH, fold, model, REAL_CHECK_DATANAME))
    pred_last = np.load(os.path.join(RESULTS_PATH, fold, model, PRED_LAST_DATANAME))
    real_last = np.load(os.path.join(RESULTS_PATH, fold, model, REAL_LAST_DATANAME))

    pred_check = moving_average(pred_check, int(ma_n))
    real_check = moving_average(real_check, int(ma_n))
    pred_last = moving_average(pred_last, int(ma_n))
    real_last = moving_average(real_last, int(ma_n))

    if checkpoint == "0":
        print("Plotando checkpoint")
        plotAudioPowerWithPrediction(real_check, pred_check)
        
    elif checkpoint == "1":
        print("Plotando last epoch")
        plotAudioPowerWithPrediction(real_last, pred_last)
