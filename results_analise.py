import pandas as pd
from os.path import join
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr 

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

NUMBER_OF_FOLDS = 10
NUMBER_OF_MODELS = 42

###############################EXCEL DE FIT HISTORY#######################################################
print('processing fit_history data')
my_csv = [None]*NUMBER_OF_FOLDS*NUMBER_OF_MODELS
for fold in range(NUMBER_OF_FOLDS):
    for model in range(NUMBER_OF_MODELS):
        my_csv[NUMBER_OF_MODELS*fold+model] = pd.read_csv(join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'fit_history.csv'))
hor_concat = [None]*NUMBER_OF_FOLDS
for i in range(NUMBER_OF_FOLDS):
    hor_concat[i] = pd.concat([my_csv[NUMBER_OF_MODELS*i+j] for j in range(NUMBER_OF_MODELS)], axis=1)

myFit_history = pd.concat([hor_concat[i] for i in range(NUMBER_OF_FOLDS)], axis=0)

myFit_history.to_excel("myFit_history.xlsx")

###############################EXCEL DE MSE#######################################################
print('calculating mse data')
my_mse = [None]*NUMBER_OF_FOLDS*NUMBER_OF_MODELS

for fold in range(NUMBER_OF_FOLDS):
    for model in range(NUMBER_OF_MODELS):
        prediction_checkpoint_path = join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'res_prediction_checkpoint_test.npy')
        real_checkpoint_path = join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'res_real_checkpoint_test.npy')
        prediction_lastepoch_path = join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'res_prediction_lastepoch_test.npy')
        real_lastepoch_path = join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'res_real_lastepoch_test.npy')

        prediction_checkpoint = np.load(prediction_checkpoint_path)
        real_checkpoint = np.load(real_checkpoint_path)
        prediction_lastepoch = np.load(prediction_lastepoch_path)
        real_lastepoch = np.load(real_lastepoch_path)

        n0_c_mse = mse(real_checkpoint, prediction_checkpoint)
        n9_c_mse = mse(moving_average(real_checkpoint, 9), moving_average(prediction_checkpoint, 9))
        n30_c_mse = mse(moving_average(real_checkpoint, 30), moving_average(prediction_checkpoint, 30))
        n90_c_mse = mse(moving_average(real_checkpoint, 90), moving_average(prediction_checkpoint, 90))

        n0_le_mse = mse(real_lastepoch, prediction_lastepoch)
        n9_le_mse = mse(moving_average(real_lastepoch, 9), moving_average(prediction_lastepoch, 9))
        n30_le_mse = mse(moving_average(real_lastepoch, 30), moving_average(prediction_lastepoch, 30))
        n90_le_mse = mse(moving_average(real_lastepoch, 90), moving_average(prediction_lastepoch, 90))

        d = {'Fold': [fold],
            'Model': [model],
            'check_mse_0':n0_c_mse,
            'check_mse_9':n9_c_mse,
            'check_mse_30':n30_c_mse,
            'check_mse_90':n90_c_mse,
            'last_mse_0':n0_le_mse,
            'last_mse_9':n9_le_mse,
            'last_mse_30':n30_le_mse,
            'last_mse_90':n90_le_mse}

        my_mse[NUMBER_OF_MODELS*fold+model] = pd.DataFrame(data=d)

myMSE = pd.concat([my_mse[i] for i in range(NUMBER_OF_FOLDS*NUMBER_OF_MODELS)], axis=0)
myMSE.to_excel("myMSE.xlsx")

###############################EXCEL DE CORRELACAO#######################################################
print('calculating pearsonr data')
my_pearsonr = [None]*NUMBER_OF_FOLDS*NUMBER_OF_MODELS

for fold in range(NUMBER_OF_FOLDS):
    for model in range(NUMBER_OF_MODELS):
        prediction_checkpoint_path = join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'res_prediction_checkpoint_test.npy')
        real_checkpoint_path = join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'res_real_checkpoint_test.npy')
        prediction_lastepoch_path = join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'res_prediction_lastepoch_test.npy')
        real_lastepoch_path = join('results-true', 'fold_'+str(fold), 'model_foldtraining_'+str(model), 'res_real_lastepoch_test.npy')

        prediction_checkpoint = np.load(prediction_checkpoint_path)
        real_checkpoint = np.load(real_checkpoint_path)
        prediction_lastepoch = np.load(prediction_lastepoch_path)
        real_lastepoch = np.load(real_lastepoch_path)

        n0_c_pearsonr, p_value = pearsonr(moving_average(real_checkpoint, 1), moving_average(prediction_checkpoint, 1))
        n9_c_pearsonr, p_value = pearsonr(moving_average(real_checkpoint, 9), moving_average(prediction_checkpoint, 9))
        n30_c_pearsonr, p_value = pearsonr(moving_average(real_checkpoint, 30), moving_average(prediction_checkpoint, 30))
        n90_c_pearsonr, p_value = pearsonr(moving_average(real_checkpoint, 90), moving_average(prediction_checkpoint, 90))

        n0_le_pearsonr, p_value = pearsonr(moving_average(real_lastepoch, 1), moving_average(prediction_lastepoch, 1))
        n9_le_pearsonr, p_value = pearsonr(moving_average(real_lastepoch, 9), moving_average(prediction_lastepoch, 9))
        n30_le_pearsonr, p_value = pearsonr(moving_average(real_lastepoch, 30), moving_average(prediction_lastepoch, 30))
        n90_le_pearsonr, p_value = pearsonr(moving_average(real_lastepoch, 90), moving_average(prediction_lastepoch, 90))

        d = {'Fold': [fold],
            'Model': [model],
            'check_pearsonr_0':n0_c_pearsonr,
            'check_pearsonr_9':n9_c_pearsonr,
            'check_pearsonr_30':n30_c_pearsonr,
            'check_pearsonr_90':n90_c_pearsonr,
            'last_pearsonr_0':n0_le_pearsonr,
            'last_pearsonr_9':n9_le_pearsonr,
            'last_pearsonr_30':n30_le_pearsonr,
            'last_pearsonr_90':n90_le_pearsonr}

        my_pearsonr[NUMBER_OF_MODELS*fold+model] = pd.DataFrame(data=d)

myPearsonr = pd.concat([my_pearsonr[i] for i in range(NUMBER_OF_FOLDS*NUMBER_OF_MODELS)], axis=0)
myPearsonr.to_excel("myPearsonr.xlsx")