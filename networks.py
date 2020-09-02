from include.globals_and_functions import *

"""
    This file contains 
    Python dictionaries with all networks to be tested 

    Initialize a python dictionary with the networks structures to be tested
   
    The basic network structure that is being tested is based on network a) of Carreira, J., & Zisserman, A. (2017). Quo Vadis, action recognition? A new model and the kinetics dataset.

    timeDistributed(VGG16) -> 
    timeDistributed(Pooling) -> 
    rcnn -> 
    Hidden FC -> 
    FC (output)

    We only change some parameters

    Variáveis a serem analisadas:

    causal_prediction
    pooling_input
    time_steps
    lstm_dropout
    fc_nlinear_activity_regularizer

"""
net_number = 32
networks = [dict() for i in range(net_number)]

#   Diferent parameters generation  --------------------------------- #

#   FIXED PARAMETERS FOR ALL NETWORKS
for i in range(net_number):
    networks[i]['model_name'] = 'model_math_lstm_'+str(i)
    networks[i]['dataset_overlap_windows'] = True
    networks[i]['overlaping_window'] = True
    networks[i]['features_input'] = True
    networks[i]['pooling_input'] = None
    networks[i]['pooling'] = 'GAP'
    networks[i]['time_steps'] = 9
    networks[i]['hidden_fc'] = True

    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['batch_size'] = 32
    networks[i]['number_of_epochs'] = 90
    networks[i]['loss_function'] = 'mse'    
    networks[i]['optimizer'] = 'adam'

    networks[i]['rcnn_type'] = 'no_rnn'
    networks[i]['lstm_units'] = 128

    networks[i]['fc_nlinear_activation'] = 'tanh'
    networks[i]['fc_nlinear_size'] = 128

for i in range(net_number):
    if i%2 < (2-1)/2:
        networks[i]['dataset_causal_prediction'] = True
    else:
        networks[i]['dataset_causal_prediction'] = False        

    if i%4 < (4-1)/2:
        networks[i]['pooling_input'] = 'GAP'
    else:
        networks[i]['pooling_input'] = 'GMP'        

    if i%8 < (8-1)/2:
        networks[i]['time_steps'] = 9
    else:
        networks[i]['time_steps'] = 17

    if i%16 < (16-1)/2:
        networks[i]['lstm_dropout'] = 0.2
    else:
        networks[i]['lstm_dropout'] = 0

    if i%32 < (32-1)/2:
        networks[i]['fc_nlinear_activity_regularizer'] = 'l2'
    else:
        networks[i]['fc_nlinear_activity_regularizer'] = None
"""

#   MATH_NOLSTM

#   FIXED PARAMETERS FOR ALL NETWORKS
for i in range(net_number):
    networks[i]['model_name'] = 'model_math_nolstm_'+str(i)
    networks[i]['dataset_overlap_windows'] = True
    networks[i]['overlaping_window'] = True
    networks[i]['features_input'] = True
    networks[i]['pooling_input'] = None
    networks[i]['time_steps'] = 9

    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['batch_size'] = 32
    networks[i]['number_of_epochs'] = 90
    networks[i]['loss_function'] = 'mse'    
    networks[i]['optimizer'] = 'adam'

    networks[i]['rcnn_type'] = 'no_rnn'
    networks[i]['lstm_units'] = 128

    networks[i]['fc_nlinear_activation'] = 'tanh'
    networks[i]['fc_nlinear_size'] = 128

for i in range(net_number):
    if i%2 < (2-1)/2:
        networks[i]['pooling'] = 'GAP'
    else:
        networks[i]['pooling'] = 'GMP'        

    if i < 2:
        networks[i]['hidden_fc'] = False
    else:
        networks[i]['hidden_fc'] = True

    if (i-2)%4 < (4-1)/2:
        networks[i]['fc_nlinear_activity_regularizer'] = None
    else:
        networks[i]['fc_nlinear_activity_regularizer'] = 'l2'

    if (i-2)%8 < (8-1)/2:
        networks[i]['fc_nlinear_activation'] = 'relu'
    else:
        networks[i]['fc_nlinear_activation'] = 'tanh'
"""
"""
#   MATH_LSTM_NOHIDDEN_i

for i in range(net_number):
    networks[i]['model_name'] = 'model_math_lstm_nohidden_'+str(i)
    networks[i]['dataset_overlap_windows'] = True
    networks[i]['overlaping_window'] = True
    networks[i]['features_input'] = True
    networks[i]['pooling'] = 'GAP'

    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['batch_size'] = 32
    networks[i]['number_of_epochs'] = 90
    networks[i]['loss_function'] = 'mse'    
    networks[i]['optimizer'] = 'adam'

    networks[i]['rcnn_type'] = 'lstm'
    networks[i]['lstm_units'] = 128

    networks[i]['hidden_fc'] = False
    networks[i]['fc_nlinear_activation'] = 'tanh'
    networks[i]['fc_nlinear_size'] = 128

for i in range(net_number):
    if i%2 < (2-1)/2:
        networks[i]['dataset_causal_prediction'] = True
    else:
        networks[i]['dataset_causal_prediction'] = False        

    if i%4 < (4-1)/2:
        networks[i]['pooling_input'] = 'GAP'
    else:
        networks[i]['pooling_input'] = 'GMP'        

    if i%8 < (8-1)/2:
        networks[i]['time_steps'] = 9
    else:
        networks[i]['time_steps'] = 17

    if i < 8:
        networks[i]['lstm_dropout'] = 0.2
    elif i < 16:
        networks[i]['lstm_dropout'] = 0
    else:
        networks[i]['lstm_dropout'] = 0.5
"""
#   Network Parameters -------------------------------------------- #
#networks[0]['dataset_causal_prediction'] = True
#networks[0]['pooling_input'] = 'GMP'
#networks[0]['time_steps'] = 9
#networks[0]['lstm_dropout'] = 0.2
#networks[0]['hidden_fc'] = True
#networks[0]['fc_nlinear_activity_regularizer'] = 'l2'
